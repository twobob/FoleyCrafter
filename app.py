import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime
import gc
import numpy as np

import gradio as gr
import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy


os.environ["GRADIO_TEMP_DIR"] = "./tmp"

sample_idx = 0
scheduler_dict = {
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="example/config/base.yaml")
parser.add_argument("--server-name", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", type=bool, default=False)

parser.add_argument("--save-path", default="samples")
parser.add_argument("--ckpt", type=str, default="checkpoints/")

args = parser.parse_args()


N_PROMPT = ""


class FoleyController:
    def __init__(self):
        # config dirs
        self.basedir = os.getcwd()
        self.model_dir = os.path.join(self.basedir, args.ckpt)
        self.savedir = os.path.join(
            self.basedir, args.save_path, datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S")
        )
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir_sample, exist_ok=True)

        self.pipeline = None

        self.loaded = False

        self.load_model()

    def load_model(self):
        print("Start Load Models...")

        # download ckpt
        pretrained_model_name_or_path = "auffusion/auffusion-full-no-adapter"
        if not os.path.isdir(osp.join(self.model_dir, "auffusion")):
            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path, local_dir=osp.join(self.model_dir, "auffusion")
            )

        fc_ckpt = "ymzhang319/FoleyCrafter"
        if not os.path.isdir(osp.join(self.model_dir, "FoleyCrafter")):
            fc_ckpt = snapshot_download(fc_ckpt, local_dir=osp.join(self.model_dir, "FoleyCrafter"))

        # set model config
        temporal_ckpt_path = osp.join(self.model_dir, "FoleyCrafter", "temporal_adapter.ckpt")

        # load vocoder
        vocoder_config_path = osp.join(self.model_dir, "auffusion")
        self.vocoder = Generator.from_pretrained(vocoder_config_path, subfolder="vocoder")

        # load time detector
        time_detector_ckpt = osp.join(self.model_dir, "FoleyCrafter", "timestamp_detector.pth.tar")
        time_detector = VideoOnsetNet(False)
        self.time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, strict=True)

        self.pipeline = build_foleycrafter()
        ckpt = torch.load(temporal_ckpt_path, map_location="cpu")

        # load temporal adapter
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        load_gligen_ckpt = {}
        for key, value in ckpt.items():
            if key.startswith("module."):
                load_gligen_ckpt[key[len("module.") :]] = value
            else:
                load_gligen_ckpt[key] = value
        m, u = self.pipeline.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
        print(f"### Control Net missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder"
        )

        self.pipeline.load_ip_adapter(
            osp.join(self.model_dir, "FoleyCrafter"),
            subfolder="semantic",
            weight_name="semantic_adapter.bin",
            image_encoder_folder=None,
        )

        print("Load Finish!")
        self.loaded = True

        return "Load"

    def foley(
        self,
        input_video,
        prompt_textbox,
        negative_prompt_textbox,
        ip_adapter_scale,
        temporal_scale,
        sampler_dropdown,
        sample_step_slider,
        cfg_scale_slider,
        seed_textbox,
        overwrite_checkbox,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device)
        if seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
            generator.manual_seed(int(seed_textbox))

        # Define the video transformation outside the with block
        vision_transform_list = [
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.CenterCrop((112, 112)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        video_transform = torchvision.transforms.Compose(vision_transform_list)

        with torch.no_grad():
            # Move models to GPU
            self.time_detector = self.time_detector.to(device)
            self.pipeline = self.pipeline.to(device)
            self.vocoder = self.vocoder.to(device)
            self.image_encoder = self.image_encoder.to(device)

            # Process video frames
            frames, duration = read_frames_with_moviepy(input_video)
            max_duration = 60  # Maximum duration
            duration = min(duration, max_duration)

            total_frames = len(frames)
            frames_per_second = total_frames / duration

            # Calculate number of chunks
            chunk_duration = 10  # Duration per chunk
            num_chunks = int(np.ceil(duration / chunk_duration))

            audio_outputs = []
            video_clips = []

            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * chunk_duration
                end_time = min((chunk_idx + 1) * chunk_duration, duration)
                chunk_duration_actual = end_time - start_time

                # Get frame indices for this chunk
                start_frame_idx = int(start_time * frames_per_second)
                end_frame_idx = int(end_time * frames_per_second)
                chunk_frames = frames[start_frame_idx:end_frame_idx]

                if len(chunk_frames) == 0:
                    continue  # Skip if no frames in this chunk

                # Process time frames
                time_frames = torch.FloatTensor(chunk_frames).permute(0, 3, 1, 2)
                time_frames = video_transform(time_frames)
                time_frames = time_frames.to(device)
                time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}
                preds = self.time_detector(time_frames)
                preds = torch.sigmoid(preds)

                # Prepare time condition
                num_audio_frames = 1024
                time_condition = [
                    -1 if preds[0][int(i / num_audio_frames * len(chunk_frames))] < 0.5 else 1
                    for i in range(num_audio_frames)
                ]
                time_condition = torch.FloatTensor(time_condition).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                time_condition = time_condition.repeat(1, 1, 256, 1).to(device)

                # Process images for image embeddings
                step = max(1, len(chunk_frames) // 10)
                frames_to_process = chunk_frames[::step]
                images = self.image_processor(images=frames_to_process, return_tensors="pt").to(device)
                image_embeddings = self.image_encoder(**images).image_embeds
                image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
                neg_image_embeddings = torch.zeros_like(image_embeddings)
                image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1).to(device)

                # Set scales and run pipeline
                self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
                sample = self.pipeline(
                    prompt=prompt_textbox,
                    negative_prompt=negative_prompt_textbox,
                    ip_adapter_image_embeds=image_embeddings,
                    image=time_condition,
                    controlnet_conditioning_scale=float(temporal_scale),
                    num_inference_steps=sample_step_slider,
                    height=256,
                    width=1024,
                    output_type="pt",
                    generator=generator,
                )

                # Process output
                audio_img = sample.images[0]
                audio = denormalize_spectrogram(audio_img)
                audio = self.vocoder.inference(audio, lengths=160000)[0]
                audio = audio[: int(chunk_duration_actual * 16000)]
                # Ensure audio is a NumPy array on CPU
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                audio_outputs.append(audio)

                # Process video clip
                video_clip = VideoFileClip(input_video).subclip(start_time, end_time)
                video_clips.append(video_clip)

                # Delete variables related to the chunk and free memory
                del time_frames, images, image_embeddings, time_condition, sample, audio_img, audio, video_clip
                torch.cuda.empty_cache()
                gc.collect()

            # Concatenate audio outputs
            if len(audio_outputs) == 0:
                return None  # Return if no audio outputs
            full_audio = np.concatenate(audio_outputs)

            # Adjust output directory and file names
            if overwrite_checkbox:
                name = "output"
                output_dir = self.savedir_sample
            else:
                name = datetime.now().strftime("%Y%m%d%H%M%S")
                output_dir = os.path.join(self.savedir_sample, name)
                os.makedirs(output_dir, exist_ok=True)

            audio_save_path = osp.join(output_dir, "audio")
            os.makedirs(audio_save_path, exist_ok=True)
            save_path = osp.join(audio_save_path, f"{name}.wav")
            sf.write(save_path, full_audio, 16000)

            # Combine audio with video
            audio_clip = AudioFileClip(save_path)
            final_video_clip = concatenate_videoclips(video_clips)
            final_video_clip.audio = audio_clip
            output_video_path = osp.join(output_dir, f"{name}.mp4")
            final_video_clip.write_videofile(output_video_path)
            save_sample_path = output_video_path

            # Move models back to CPU
            self.time_detector.to("cpu")
            self.pipeline.to("cpu")
            self.vocoder.to("cpu")
            self.image_encoder.to("cpu")

            # Clear GPU memory and collect garbage
            torch.cuda.empty_cache()
            gc.collect()

        return save_sample_path


controller = FoleyController()
device = "cuda" if torch.cuda.is_available() else "cpu"

with gr.Blocks(css=css) as demo:
    gr.HTML(
        '<h1 style="height: 136px; display: flex; align-items: center; justify-content: space-around;"><span style="height: 100%; width:136px;"><img src="./assets/foleycrafter.png" alt="logo" style="height: 100%; width:auto; object-fit: contain; margin: 0px 0px; padding: 0px 0px;"></span><strong style="font-size: 36px;">FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds</strong></h1>'
    )

    with gr.Column(variant="panel"):
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    init_img = gr.Video(label="Input Video")
                with gr.Row():
                    prompt_textbox = gr.Textbox(value="", label="Prompt", lines=1)
                with gr.Row():
                    negative_prompt_textbox = gr.Textbox(value=N_PROMPT, label="Negative prompt", lines=1)

                with gr.Row():
                    ip_adapter_scale = gr.Slider(label="Visual Content Scale", value=1.0, minimum=0, maximum=1)
                    temporal_scale = gr.Slider(label="Temporal Align Scale", value=0.2, minimum=0.0, maximum=1.0)

                with gr.Accordion("Sampling Settings", open=False):
                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(
                            label="Sampling method",
                            choices=list(scheduler_dict.keys()),
                            value=list(scheduler_dict.keys())[0],
                        )
                        sample_step_slider = gr.Slider(
                            label="Sampling steps", value=25, minimum=10, maximum=100, step=1
                        )
                    cfg_scale_slider = gr.Slider(label="CFG Scale", value=7.5, minimum=0, maximum=20)

                with gr.Row():
                    seed_textbox = gr.Textbox(label="Seed", value=1337)
                    seed_button = gr.Button(value="\U0001f3b2", elem_classes="toolbutton")
                seed_button.click(fn=lambda x: random.randint(1, 1e8), outputs=[seed_textbox], queue=False)

                with gr.Row():
                    overwrite_checkbox = gr.Checkbox(label="Overwrite existing outputs", value=True)

                generate_button = gr.Button(value="Generate", variant="primary")

            with gr.Column():
                result_video = gr.Video(label="Generated Audio", interactive=False)
                with gr.Row():
                    gr.Markdown(
                        "<div style='word-spacing: 6px;'><font size='5'><b>Tips</b>: <br> \
                        1. With strong temporal visual cues in input video, you can scale up the <b>Temporal Align Scale</b>. <br>\
                        2. <b>Visual content scale</b> is the level of semantic alignment with visual content.</font></div> \
                    "
                    )

        generate_button.click(
            fn=controller.foley,
            inputs=[
                init_img,
                prompt_textbox,
                negative_prompt_textbox,
                ip_adapter_scale,
                temporal_scale,
                sampler_dropdown,
                sample_step_slider,
                cfg_scale_slider,
                seed_textbox,
                overwrite_checkbox,
            ],
            outputs=[result_video],
        )

        gr.Examples(
            examples=[
                [
                    "examples/gen3/case1.mp4",
                    "bear talking to camera in childrens show",
                    "",
                    1.0,
                    0.15,
                    "DDIM",
                    45,
                    7.5,
                    12131415,
                    True,
                ],
                [
                    "examples/gen3/case3.mp4",
                    "fly swatter misses fly in slow motion",
                    "",
                    1.0,
                    0.2,
                    "DDIM",
                    45,
                    7.5,
                    45148336,
                    True,
                ],
                [
                    "examples/gen3/case5.mp4",
                    "lava eruption over volcano in a helicopter",
                    "",
                    0.75,
                    0.2,
                    "DDIM",
                    45,
                    7.5,
                    1984,
                    True,
                ],
                [
                    "examples/gen3/case6.mp4",
                    "stuffed puppet vocalises in childrens television show",
                    "",
                    1.0,
                    0.3,
                    "DDIM",
                    45,
                    7.5,
                    1337,
                    True,
                ],
            ],
            inputs=[
                init_img,
                prompt_textbox,
                negative_prompt_textbox,
                ip_adapter_scale,
                temporal_scale,
                sampler_dropdown,
                sample_step_slider,
                cfg_scale_slider,
                seed_textbox,
                overwrite_checkbox,
            ],
            cache_examples=True,
            outputs=[result_video],
            fn=controller.foley,
        )

    demo.queue(10)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        allowed_paths=["./assets/foleycrafter.png"],
    )
