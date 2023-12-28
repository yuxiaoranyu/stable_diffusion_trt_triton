import os
from collections import OrderedDict
from copy import copy
from typing import List, Optional, Union
import numpy as np
import PIL.Image
import tensorrt as trt
import torch
from huggingface_hub import snapshot_download
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util
from transformers import CLIPTokenizer
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import DIFFUSERS_CACHE, logging

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def device_view(t):
    return cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=torch_to_numpy_dtype_dict[t.dtype])

class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        workspace_size=0,
    ):
        logger.warning(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs["preview_features"] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs["preview_features"].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(fp16=fp16, profiles=[p], load_timing_cache=timing_cache, **config_kwargs),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.warning(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


class BaseModel:
    def __init__(self, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen


    def get_shape_dict(self, batch_size, image_height, image_width):
        return None


    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)


def getEnginePath(model_name, engine_dir):
    return os.path.join(engine_dir, model_name + ".plan")


def build_engines(engine_dir):
    built_engines = {}
    model_names = ["clip", "unet", "vae", "vae_encoder"]
    for model_name in model_names:
        print("model_name: ", model_name)
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        built_engines[model_name] = engine
        engine.load()
        engine.activate()

    return built_engines


def runEngine(engine, feed_dict, stream):
    return engine.infer(feed_dict, stream)


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim):
        super(CLIP, self).__init__(
            device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "CLIP"

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }



def make_CLIP(device, max_batch_size, embedding_dim, inpaint=False):
    return CLIP(device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class UNet(BaseModel):
    def __init__(
        self, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77, unet_dim=4
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }


def make_UNet(device, max_batch_size, embedding_dim, inpaint=False, unet_dim=4):
    return UNet(
        fp16=True,
        device=device,
        max_batch_size=max_batch_size,
        embedding_dim=embedding_dim,
        unet_dim=unet_dim,
    )


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim):
        super(VAE, self).__init__(
            device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "VAE decoder"

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }


def make_VAE(device, max_batch_size, embedding_dim, inpaint=False):
    return VAE(device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim):
        super(VAEEncoder, self).__init__(
            device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "VAE encoder"


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }


def make_VAEEncoder(device, max_batch_size, embedding_dim, inpaint=False):
    return VAEEncoder(device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)



class TensorRTStableDiffusionInpaintPipeline():
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler: DDIMScheduler,
        stages=["clip", "unet", "vae", "vae_encoder"],
        engine_dir: str = "engine",
    ):
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.stages = stages
        self.engine_dir = engine_dir
        self.inpaint = True
        self.max_batch_size = 1
        self.stream = None  # loaded in loadResources()
        self.models = {}  # loaded in __loadModels()
        self.engine = {}  # loaded in build_engines()

    def __loadModels(self):
        # Load pipeline models
        # self.embedding_dim = self.text_encoder.config.hidden_size
        self.embedding_dim = 768
        models_args = {
            "device": self.torch_device,
            "max_batch_size": self.max_batch_size,
            "embedding_dim": self.embedding_dim,
            "inpaint": self.inpaint,
        }
        if "clip" in self.stages:
            self.models["clip"] = make_CLIP(**models_args)
        if "unet" in self.stages:
            self.models["unet"] = make_UNet(**models_args, unet_dim=9)
        if "vae" in self.stages:
            self.models["vae"] = make_VAE(**models_args)
        if "vae_encoder" in self.stages:
            self.models["vae_encoder"] = make_VAEEncoder(**models_args)

    @classmethod
    def set_cached_folder(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
    
        cls.cached_folder = (
            pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path)
            else snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        )
    def numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [PIL.Image.fromarray(image) for image in images]

        return pil_images
    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings: bool = False):
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)

        # set device
        self.torch_device = torch_device
        logger.warning(f"Running inference on device: {self.torch_device}")

        # load models
        self.__loadModels()
        self.engine = build_engines(self.engine_dir)
        # print("self.image_height: ", self.image_height)
        # load resources
        self.__loadResources(self.image_height, self.image_width, 1)
        return self

    def __initialize_timesteps(self, num_inference_steps, strength):
        self.scheduler.set_timesteps(num_inference_steps)
        offset = self.scheduler.config.steps_offset if hasattr(self.scheduler, "steps_offset") else 0
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :].to(self.torch_device)
        return timesteps, num_inference_steps - t_start

    def __preprocess_images(self, batch_size, images=()):
        init_images = []
        for image in images:
            image = image.to(self.torch_device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        return tuple(init_images)

    def __encode_image(self, init_image):
        init_latents = runEngine(self.engine["vae_encoder"], {"images": device_view(init_image)}, self.stream)[
            "latent"
        ]
        init_latents = 0.18215 * init_latents
        return init_latents

    def __encode_prompt(self, prompt, negative_prompt):
        text_input_ids = (
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        text_input_ids_inp = device_view(text_input_ids)
        text_embeddings = runEngine(self.engine["clip"], {"input_ids": text_input_ids_inp}, self.stream)[
            "text_embeddings"
        ].clone()

        # Tokenize negative prompt
        uncond_input_ids = (
            self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )
        uncond_input_ids_inp = device_view(uncond_input_ids)
        uncond_embeddings = runEngine(self.engine["clip"], {"input_ids": uncond_input_ids_inp}, self.stream)[
            "text_embeddings"
        ]

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings

    def __denoise_latent(
        self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None
    ):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = runEngine(
                self.engine["unet"],
                {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp},
                self.stream,
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        latents = 1.0 / 0.18215 * latents
        return latents
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )
        print("return_image_latents: ", return_image_latents, is_strength_max, latents)
        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            print("image: ", image.shape)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs
    
    def __decode_latent(self, latents):
        images = runEngine(self.engine["vae"], {"latent": device_view(latents)}, self.stream)["images"]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()

    def __loadResources(self, image_height, image_width, batch_size):
        self.stream = cuda.Stream()

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            shape_dict=obj.get_shape_dict(batch_size, image_height, image_width)
            print("{} shape_dict: ".format(model_name), shape_dict)
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.torch_device
            )
        del self.models

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        self.generator = generator
        self.denoising_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps, device=self.torch_device)

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Expected prompt to be of type list or str but got {type(prompt)}")
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompt)} is larger than allowed {self.max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
            )
       
        # Validate image dimensions
        mask_width, mask_height = mask_image.size
        if mask_height != self.image_height or mask_width != self.image_width:
            raise ValueError(
                f"Input image height and width {self.image_height} and {self.image_width} are not equal to "
                f"the respective dimensions of the mask image {mask_height} and {mask_width}"
            )
        print("self.image_height: ", self.image_height, self.image_width, batch_size)
        # load resources
        # self.__loadResources(self.image_height, self.image_width, batch_size)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Spatial dimensions of latent tensor
            latent_height = self.image_height // 8
            latent_width = self.image_width // 8

            # Pre-process input images
            mask, masked_image, init_image = self.__preprocess_images(
                batch_size,
                prepare_mask_and_masked_image(
                    image,
                    mask_image,
                    self.image_height,
                    self.image_width,
                    return_image=True,
                ),
            )

            mask = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width))
            mask = torch.cat([mask] * 2)

            # Initialize timesteps
            timesteps, t_start = self.__initialize_timesteps(self.denoising_steps, strength)

            # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
            latent_timestep = timesteps[:1].repeat(batch_size)
            # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
            is_strength_max = strength == 1.0

            # Pre-initialize latents
            # num_channels_latents = self.vae.config.latent_channels
            num_channels_latents = 4
            latents_outputs = self.prepare_latents(
                batch_size,
                num_channels_latents,
                self.image_height,
                self.image_width,
                torch.float32,
                self.torch_device,
                generator,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
            )

            latents = latents_outputs[0]

            # VAE encode masked image
            masked_latents = self.__encode_image(masked_image)
            masked_latents = torch.cat([masked_latents] * 2)

            # CLIP text encoder
            text_embeddings = self.__encode_prompt(prompt, negative_prompt)

            # UNet denoiser
            latents = self.__denoise_latent(
                latents,
                text_embeddings,
                timesteps=timesteps,
                step_offset=t_start,
                mask=mask,
                masked_image_latents=masked_latents,
            )
            # VAE decode latent
            images = self.__decode_latent(latents)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


if __name__=="__main__":
    from diffusers.utils import load_image
    inpaint_model_path="/workspace/code/stable_diffusion_inpainting/"
    tokenizer = CLIPTokenizer.from_pretrained(inpaint_model_path+"tokenizer")
    scheduler = DDIMScheduler.from_pretrained(inpaint_model_path+"scheduler")
    trt_inpaint_pipe=TensorRTStableDiffusionInpaintPipeline(
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    trt_inpaint_pipe.set_cached_folder(inpaint_model_path)
    size = 832
    trt_inpaint_pipe.image_width = size
    trt_inpaint_pipe.image_height = size
    device = torch.device('cuda')
    trt_inpaint_pipe.to(device)
    init_image_path = "image/sugar.jpg"
    mask_image_path = "image/cp.jpg"
    
    init_image = load_image(init_image_path).resize((size,size))
    mask_image = load_image(mask_image_path).resize((size,size))
    print("init_image: ", np.array(init_image).shape)
    print("ori_mask: ", np.array(mask_image).shape)
    generator = torch.Generator(device).manual_seed(54924510)
    import time
    time_list=[]
    for i in range(1):
        start = time.time()
        image = trt_inpaint_pipe(
            prompt="a few maple leaf scattered around, cozy day, outdoors, nature . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            generator = generator,
            negative_prompt='',
            image=init_image,
            mask_image=mask_image,
        ).images[0]
        
        print("time cost: {} ms".format((time.time()-start)*1000))
        time_list.append((time.time()-start)*1000)
    print("aver age time {} ms:".format(sum(time_list)/len(time_list)))
    image.save("1.png")