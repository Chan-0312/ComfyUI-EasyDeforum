
from nodes import VAEDecode, VAEEncode
import comfy
from .utils import image_2dtransform, apply_easing, easing_functions
import torch
from tqdm import tqdm





class Easy2DDeforum:
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "vae": ("VAE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "frame": ("INT", {"default": 16}),
                "x": ("INT", {"default": 15, "step": 1, "min": -4096, "max": 4096}),
                "y": ("INT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 0.98, "min": 0.001, "step": 0.01}),
                "angle": ("INT", {"default": -1, "step": 1, "min": -360, "max": 360}),
                "denoise_min": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step":0.01}),
                "denoise_max": ("FLOAT", {"default": 0.60, "min": 0.00, "max": 1.00, "step":0.01}),
                "easing_type": (list(easing_functions.keys()), ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "apply"

    CATEGORY = "easyDeforum"

    def apply(self, model, vae, positive, negative, image, frame, seed, steps, cfg, sampler_name, scheduler, x, y, zoom, angle, denoise_min, denoise_max, easing_type):
        
        # 初始化模型
        vaedecode = VAEDecode()
        vaeencode = VAEEncode()

        res = [image]

        pbar = comfy.utils.ProgressBar(frame)
        for i in tqdm(range(frame)):
            
            # 计算噪声
            denoise = (denoise_max - denoise_min) * apply_easing((i+1)/frame, easing_type)  + denoise_min

            # 变换
            image = image_2dtransform(image, x, y, zoom, angle, 0, "reflect")

            # 编码
            latent = vaeencode.encode(vae, image)[0]


            # 计算噪声
            noise = comfy.sample.prepare_noise(latent["samples"], i, None)

            # 采样
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                        denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                        force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)
        
            # 解码
            image = vaedecode.decode(vae, {"samples": samples})[0]

            # 这里还能加预览图片
            pbar.update_absolute(i + 1, frame, None)

            res.append(image)

        # 如果第一张图片和生成的图片尺寸不一致，则丢弃
        if res[0].size() != res[-1].size():
            res = res[1:]

        res = torch.cat(res, dim=0)
        return (res, )


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Easy2DDeforum": Easy2DDeforum,
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "Easy2DDeforum": "Easy2DDeforum (Chan)"
}
