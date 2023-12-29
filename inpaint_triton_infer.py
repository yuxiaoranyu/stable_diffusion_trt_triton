import os
import torch
import PIL.Image
import numpy as np
from copy import copy
import tensorrt as trt
from polygraphy import cuda
from collections import OrderedDict
from transformers import CLIPTokenizer
from typing import List, Optional, Union
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.backend.trt import util as trt_util
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging
import tritonclient.http as httpclient
# import triton_python_backend_utils as pb_utils  # type: ignore

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
torch_to_numpy_dtype_dict = {value: key for (
    key, value) in numpy_to_torch_dtype_dict.items()}

def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image

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
        [buf.free() for buf in self.buffers.values()
         if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors


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
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(
                ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(
            self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + \
            [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(
            bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


def getEnginePath(model_name, engine_dir):
    return os.path.join(engine_dir, model_name + ".plan")


def build_engines(engine_dir):
    built_engines = {}
    model_name = "unet"
    engine_path = os.path.join(engine_dir, model_name + ".plan")
    engine = Engine(engine_path)
    built_engines[model_name] = engine
    engine.load()
    engine.activate()
    return built_engines


def runEngine(engine, feed_dict, stream):
    return engine.infer(feed_dict, stream)

class TensorRTStableDiffusionInpaintPipeline():
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler: DDIMScheduler,
        image_height: int = 512,
        image_width: int = 512,
        max_batch_size: int = 1,
        engine_dir: str = "engine",
    ):
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.image_height, self.image_width = image_height, image_width
        self.inpaint = True
        self.engine_dir = engine_dir
        
        self.max_batch_size = max_batch_size
        # TODO: Restrict batch size to 4 for larger image dimensions as a WAR for TensorRT limitation.
        if self.image_height > 512 or self.image_width > 512:
            self.max_batch_size = 4

        self.stream = cuda.Stream()  # loaded in loadResources()
        self.models = {}  # loaded in __loadModels()
        self.engine = {}  # loaded in build_engines()
        self.triton_client = httpclient.InferenceServerClient(
            url="localhost:8000", verbose=False)

    def numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [PIL.Image.fromarray(
                image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [PIL.Image.fromarray(image) for image in images]

        return pil_images

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings: bool = False):

        # set device
        self.torch_device = torch_device
        logger.warning(f"Running inference on device: {self.torch_device}")
        self.engine = build_engines(self.engine_dir)
        latent_height = self.image_height // 8
        latent_width = self.image_width // 8
        unet_shape_dict = {'sample': (2, 9, latent_height, latent_width), 'encoder_hidden_states': (
            2, 77, 768), 'latent': (2, 4, latent_height, latent_width)}
        self.engine["unet"].allocate_buffers(shape_dict=unet_shape_dict, device=self.torch_device)
        return self

    def __initialize_timesteps(self, num_inference_steps, strength):
        self.scheduler.set_timesteps(num_inference_steps)
        offset = self.scheduler.config.steps_offset if hasattr(
            self.scheduler, "steps_offset") else 0
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start *
                                             self.scheduler.order:].to(self.torch_device)
        return timesteps, num_inference_steps - t_start

    def __preprocess_images(self, batch_size, images=()):
        init_images = []
        for image in images:
            image = image.to(self.torch_device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        return tuple(init_images)

    def triton_encode_image(self, init_image):
        vae_encoder_inputs = []
        vae_encoder_inputs.append(httpclient.InferInput(
            "images", init_image.shape, "FP32").set_data_from_numpy(init_image.cpu().numpy(), binary_data=False))
        vae_encoder_outputs = []
        vae_encoder_outputs.append(httpclient.InferRequestedOutput('latent'))
        vae_encoder_result = self.triton_client.infer(
            "vae_encoder", vae_encoder_inputs, outputs=vae_encoder_outputs)
        init_latents = vae_encoder_result.as_numpy("latent")
        init_latents = torch.from_numpy(init_latents).to("cuda")
        init_latents = 0.18215 * init_latents
        return init_latents

    def triton_encode_prompt(self, prompt, negative_prompt):
        # Tokenize prompt
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

        # triton infer
        text_encoder_inputs = []
        text_encoder_inputs.append(httpclient.InferInput(
            'input_ids', text_input_ids_inp.shape, "INT32"))
        text_encoder_inputs[0].set_data_from_numpy(
            text_input_ids_inp.numpy().astype(np.int32), binary_data=False)
        text_encoder_outputs = []
        text_encoder_outputs.append(
            httpclient.InferRequestedOutput('text_embeddings'))
        text_encoder_result = self.triton_client.infer(
            "text_encoder", text_encoder_inputs, outputs=text_encoder_outputs)
        text_embeddings = text_encoder_result.as_numpy("text_embeddings")
        text_embeddings = torch.from_numpy(text_embeddings).to("cuda").clone()
        text_encoder_inputs.clear()
        text_encoder_outputs.clear()

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
        # triton infer
        text_encoder_inputs.append(httpclient.InferInput(
            'input_ids', uncond_input_ids_inp.shape, "INT32"))
        text_encoder_inputs[0].set_data_from_numpy(
            uncond_input_ids_inp.numpy().astype(np.int32), binary_data=False)
        text_encoder_outputs.append(
            httpclient.InferRequestedOutput('text_embeddings'))
        text_encoder_result = self.triton_client.infer(
            "text_encoder", text_encoder_inputs, outputs=text_encoder_outputs)
        uncond_embeddings = text_encoder_result.as_numpy("text_embeddings")
        uncond_embeddings = torch.from_numpy(uncond_embeddings).to("cuda")
        text_embeddings = torch.cat(
            [uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings

    def triton_denoise_latent(
        self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None
    ):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat(
                    [latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            timestep_inp = np.array([timestep_inp.numpy()])
            unet_inputs = []
            
            unet_inputs.append(pb_utils.Tensor('sample', latent_model_input.cpu().numpy()))
            unet_inputs.append(pb_utils.Tensor('timestep', timestep_inp))
            unet_inputs.append(pb_utils.Tensor('encoder_hidden_states', text_embeddings.cpu().numpy()))

            inference_request = pb_utils.InferenceRequest(
                        model_name='unet',
                        requested_output_names=['latent'],
                        inputs=unet_inputs
                    )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, 'latent'
                )
                noise_pred: torch.Tensor = torch.from_dlpack(output.to_dlpack())

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, timestep, latents).prev_sample

        latents = 1.0 / 0.18215 * latents
        return latents

    def __denoise_latent(
        self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None
    ):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat(
                    [latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = runEngine(
                self.engine["unet"],
                {"sample": sample_inp, "timestep": timestep_inp,
                    "encoder_hidden_states": embeddings_inp},
                self.stream,
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, timestep, latents).prev_sample

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

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            image_latents = image_latents.repeat(
                batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator,
                                 device=device, dtype=dtype)
            latents = noise if is_strength_max else self.scheduler.add_noise(
                image_latents, noise, timestep)
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

        outputs = (latents,)
        if return_noise:
            outputs += (noise,)
        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def triton_decode_latent(self, latents):
        vae_decoder_inputs = []
        vae_decoder_inputs.append(httpclient.InferInput(
            "latent", latents.shape, "FP32").set_data_from_numpy(latents.cpu().numpy(), binary_data=False))
        vae_decoder_outputs = []
        vae_decoder_outputs.append(httpclient.InferRequestedOutput('images'))
        vae_decoder_result = self.triton_client.infer(
            "vae_decoder", vae_decoder_inputs, outputs=vae_decoder_outputs)
        images = vae_decoder_result.as_numpy("images")
        images = torch.from_numpy(images).to("cuda")
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()

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
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
    ):
        self.generator = generator
        self.denoising_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self.scheduler.set_timesteps(
            self.denoising_steps, device=self.torch_device)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"Expected prompt to be of type list or str but got {type(prompt)}")

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

            mask = torch.nn.functional.interpolate(
                mask, size=(latent_height, latent_width))
            mask = torch.cat([mask] * 2)
            timesteps, t_start = self.__initialize_timesteps(
                self.denoising_steps, strength)
            latent_timestep = timesteps[:1].repeat(batch_size)
            is_strength_max = strength == 1.0

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
            masked_latents = self.triton_encode_image(masked_image)
            masked_latents = torch.cat([masked_latents] * 2)

            # CLIP text encoder
            text_embeddings = self.triton_encode_prompt(
                prompt, negative_prompt)

            # # triton unet infer
            # latents = self.triton_denoise_latent(
            #     latents,
            #     text_embeddings,
            #     timesteps=timesteps,
            #     step_offset=t_start,
            #     mask=mask,
            #     masked_image_latents=masked_latents,
            # )

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
            images = self.triton_decode_latent(latents)
            # images = self.__decode_latent(latents)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


if __name__ == "__main__":
    from diffusers.utils import load_image
    inpaint_model_path="/workspace/code/stable_diffusion_inpainting/"
    tokenizer = CLIPTokenizer.from_pretrained(inpaint_model_path+"tokenizer")
    scheduler = DDIMScheduler.from_pretrained(inpaint_model_path+"scheduler")
    trt_inpaint_pipe = TensorRTStableDiffusionInpaintPipeline(
        tokenizer=tokenizer,
        scheduler=scheduler,
        engine_dir=inpaint_model_path+"engine"
    )
    device = torch.device('cuda')
    trt_inpaint_pipe.to(device)
    init_image_path = "image/sugar.jpg"
    mask_image_path = "image/cp.jpg"
    size = 512
    trt_inpaint_pipe.image_width = size
    trt_inpaint_pipe.image_height = size
    init_image = load_image(init_image_path).resize((size, size))
    mask_image = load_image(mask_image_path).resize((size, size))

    generator = torch.Generator(device).manual_seed(54924510)
    import time
    time_list = []
    for i in range(1):
        start = time.time()
        image = trt_inpaint_pipe(
            prompt="a few maple leaf scattered around, cozy day, outdoors, nature . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            generator=generator,
            negative_prompt='',
            image=init_image,
            mask_image=mask_image,
        ).images[0]

        print("time cost: {} ms".format((time.time()-start)*1000))
        time_list.append((time.time()-start)*1000)
    print("aver age time {} ms:".format(sum(time_list)/len(time_list)))
    image.save("1.png")
