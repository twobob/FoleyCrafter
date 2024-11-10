import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime
import gc
import numpy as np
from tqdm import tqdm

import gradio as gr
import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    BitsAndBytesConfig,
    CLIPTokenizer,
)

from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter

# Additional imports for idefics2-8b
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import shutil  # Import shutil for file operations

os.environ["GRADIO_TEMP_DIR"] = "./tmp"

sample_idx = 0
scheduler_dict = {
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

css = """
.toolbutton {
    margin-bottom: 0em 0em 0em 0em;
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

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load idefics2-8b model and processor
        self.vision_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            device_map="auto",
            _attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
        )
        self.vision_model.eval()

        # Initialize CLIP tokenizer
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        print("Load Finish!")
        self.loaded = True

        return "Load"

    def read_frames_with_moviepy(self, video_path, max_frame_nums=None):
        frames = []
        video = VideoFileClip(video_path)
        duration = video.duration
        for frame in video.iter_frames():
            frames.append(frame)
        frames = np.array(frames)
        if max_frame_nums is not None:
            frames_idx = np.linspace(0, len(frames) - 1, max_frame_nums, dtype=int)
            frames = frames[frames_idx, ...]
        return frames, duration

    def generate_prompt_from_images(self, images, user_prompt):
        messages = []
        assistant_replies = []
        images_list = []
        seen_responses = set()  # Track unique responses

        def clean_response(text):
            # List of phrases to remove (including variations and synonyms)
            phrases_to_remove = [
                "in the image",
                "in this image",
                "in the picture",
                "in this picture",
                "pictured",
                "shown",
                "visible",
                "can be seen",
                "appears to be",
                "appearing",
                "displayed",
                "depicted",
                "present",
                "captured",
                "photographed",
                "shown here",
                "seen here",
                "in the scene",
                "in this scene",
                "in the photo",
                "in this photo",
                "in the photograph",
                "in this photograph",
                "in the shot",
                "in this shot",
                "in view",
                "in the frame",
                "in this frame",
                "Silent film",
                "Nothing.",
            ]

            # Convert to lower case for case-insensitive replacement
            text_lower = text.lower()

            # Remove each phrase and any extra spaces
            for phrase in phrases_to_remove:
                text_lower = text_lower.replace(phrase, "")

            # Clean up any multiple spaces and trim
            cleaned_text = " ".join(text_lower.split())

            # Capitalize first letter
            if cleaned_text:
                cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]

            return cleaned_text

        def generate_response(prompt_text, current_images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=current_images, return_tensors="pt")
            inputs = {k: v.to(self.vision_device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    min_length=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            for line in generated_text.strip().split("\n"):
                if line.startswith("Assistant:"):
                    response = line.replace("Assistant:", "").strip()
                    return clean_response(response)
            return ""

        for idx, image in enumerate(images):
            images_list = [image]

            # Initial prompt for sound description - removed image references
            initial_prompt = (
                "Describe briefly (under 10 words) the actions causing noises, "
                "Focus on sound, No Visual descriptions"
            ) if idx == 0 else "What would we hear?"

            response = generate_response(initial_prompt, images_list)

            # If response indicates no sound or is empty, try ambient sound prompt
            if response.lower() in [
                "nothing",
                "there is no sound",
                "",
                "answering does not require reading text",
                "unknown",
                "I can't describe a picture that shows sound.",
                "I can't describe sound but i can describe visual.",
                "I can't describe it in words.",
            ]:
                ambient_prompt = (
                    "Describe what ambient sounds would be present in this location "
                    "(wind, echoes, distant traffic, etc)"
                )
                response = generate_response(ambient_prompt, images_list)

            # Limit response to 25 words
            response = " ".join(response.split()[:25])

            # Only add if it's a unique response
            if response.lower() not in seen_responses:
                seen_responses.add(response.lower())
                assistant_replies.append(response)

            if len(assistant_replies) >= 5:
                break

        # Combine responses into final prompt
        generated_prompt = " ".join(assistant_replies)

        # Combine with user prompt if provided
        combined_prompt = f"{user_prompt} {generated_prompt}" if user_prompt else generated_prompt

        # Handle token limit
        tokens = self.clip_tokenizer.tokenize(combined_prompt)
        if len(tokens) > 77:
            user_prompt_tokens = self.clip_tokenizer.tokenize(user_prompt) if user_prompt else []
            max_generated_tokens = 77 - len(user_prompt_tokens)

            if max_generated_tokens <= 0:
                combined_prompt = self.clip_tokenizer.convert_tokens_to_string(user_prompt_tokens[:77])
            else:
                generated_prompt_tokens = self.clip_tokenizer.tokenize(generated_prompt)
                truncated_generated_prompt = self.clip_tokenizer.convert_tokens_to_string(
                    generated_prompt_tokens[:max_generated_tokens]
                )
                combined_prompt = (
                    f"{user_prompt} {truncated_generated_prompt}" if user_prompt else truncated_generated_prompt
                )

        print("\n", combined_prompt)
        return combined_prompt.strip()

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
        use_vision_model_checkbox,
        chunk_duration_slider,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device)
        if seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
            generator.manual_seed(int(seed_textbox))

        vision_transform_list = [
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.CenterCrop((112, 112)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        video_transform = torchvision.transforms.Compose(vision_transform_list)

        # Extract the video path from the input_video
        if isinstance(input_video, dict):
            temp_video_path = input_video["name"]
        else:
            temp_video_path = input_video

        # Save the uploaded video to a persistent location
        # Define a path to save the video
        saved_video_dir = os.path.join(self.savedir_sample, "input_videos")
        os.makedirs(saved_video_dir, exist_ok=True)
        video_filename = os.path.basename(temp_video_path)
        video_path = os.path.join(saved_video_dir, video_filename)

        # Copy the video from the temporary path to the persistent location
        shutil.copy(temp_video_path, video_path)

        with torch.no_grad():
            # Move models to GPU when needed
            self.time_detector = self.time_detector.to(device)
            self.pipeline = self.pipeline.to(device)
            self.vocoder = self.vocoder.to(device)
            self.image_encoder = self.image_encoder.to(device)

            # Process video frames
            frames, duration = self.read_frames_with_moviepy(video_path)
            max_duration = 600  # Maximum duration set to 5 minutes
            duration = min(duration, max_duration)

            total_frames = len(frames)
            frames_per_second = total_frames / duration

            # Calculate number of chunks
            chunk_duration = chunk_duration_slider  # Duration per chunk
            num_chunks = int(np.ceil(duration / chunk_duration))

            audio_outputs = []
            video_clips = []
            prev_latents = None

            progress_bar = tqdm(total=num_chunks, desc="Processing video chunks", unit="chunk")

            # Set up padding calculation based on actual frame count needed per chunk
            target_frame_count = int(chunk_duration * frames_per_second)  # Required frames per chunk

            try:
                for chunk_idx in range(num_chunks):
                    start_time = chunk_idx * chunk_duration
                    end_time = min((chunk_idx + 1) * chunk_duration, duration)
                    chunk_duration_actual = end_time - start_time

                    # Get frame indices for this chunk
                    start_frame_idx = int(start_time * frames_per_second)
                    end_frame_idx = int(end_time * frames_per_second)
                    chunk_frames = frames[start_frame_idx:end_frame_idx]

                    # Calculate padding for the final chunk
                    if chunk_idx == num_chunks - 1 and len(chunk_frames) < target_frame_count:
                        padding_needed = target_frame_count - len(chunk_frames)
                        print(
                            f"Padding final chunk with {padding_needed} blank frames to reach {target_frame_count} frames."
                        )
                        if len(chunk_frames) > 0:
                            height, width, channels = chunk_frames[0].shape
                        else:
                            # Default dimensions if no frames are available
                            height, width, channels = 256, 256, 3
                        blank_frame = np.zeros((height, width, channels), dtype=frames.dtype)
                        chunk_frames = np.concatenate([chunk_frames, [blank_frame] * padding_needed], axis=0)

                    # Set `combined_prompt` based on the user option
                    if use_vision_model_checkbox:
                        frame_interval = max(1, int(2 * frames_per_second))
                        idefics_frames = chunk_frames[::frame_interval]
                        idefics_images = [Image.fromarray(frame) for frame in idefics_frames]
                        combined_prompt = self.generate_prompt_from_images(
                            idefics_images, prompt_textbox.strip()
                        )
                    else:
                        combined_prompt = prompt_textbox.strip()

                    # Ensure time_frames has shape (1, 3, num_frames, 112, 112)
                    time_frames = torch.FloatTensor(chunk_frames).permute(0, 3, 1, 2)
                    time_frames = video_transform(time_frames).to(device)
                    time_frames = time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

                    # Run time detector with error handling for specific shape issues
                    try:
                        inputs = {"frames": time_frames}
                        preds = self.time_detector(inputs)
                        preds = torch.sigmoid(preds)

                        # Prepare time condition based on preds
                        num_audio_frames = 1024
                        time_condition = [
                            -1
                            if preds[0, int(i / num_audio_frames * len(chunk_frames))].item() < 0.5
                            else 1
                            for i in range(num_audio_frames)
                        ]

                    except Exception as e:
                        print(f"Warning: Time detector failed with error {str(e)}. Using default values.")
                        time_condition = [0] * 1024

                    # Convert time_condition to tensor
                    time_condition = (
                        torch.FloatTensor(time_condition).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    )
                    time_condition = time_condition.repeat(1, 1, 256, 1).to(device)

                    # Process images for image embeddings
                    step = max(1, len(chunk_frames) // 10)
                    frames_to_process = chunk_frames[::step]
                    images = self.image_processor(images=frames_to_process, return_tensors="pt").to(device)
                    image_embeddings = self.image_encoder(**images).image_embeds
                    image_embeddings = (
                        torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
                    )
                    neg_image_embeddings = torch.zeros_like(image_embeddings)
                    image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1).to(device)

                    # Set scales and configure pipeline
                    self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
                    self.pipeline.scheduler = scheduler_dict[sampler_dropdown].from_config(
                        self.pipeline.scheduler.config
                    )
                    self.pipeline.scheduler.config.steps_offset = 1
                    self.pipeline.scheduler.set_timesteps(sample_step_slider)

                    # Prepare latents
                    height = 256
                    width = 1024
                    latents_shape = (1, self.pipeline.unet.in_channels, height // 8, width // 8)
                    latents = (
                        torch.randn(latents_shape, generator=generator, device=device)
                        if prev_latents is None
                        else prev_latents
                    )

                    # Run the pipeline
                    sample = self.pipeline(
                        prompt=combined_prompt,
                        negative_prompt=negative_prompt_textbox,
                        ip_adapter_image_embeds=image_embeddings,
                        image=time_condition,
                        controlnet_conditioning_scale=float(temporal_scale),
                        num_inference_steps=sample_step_slider,
                        height=height,
                        width=width,
                        output_type="pt",
                        generator=generator,
                        latents=latents,
                        guidance_scale=cfg_scale_slider,
                        return_dict=True,
                    )

                    prev_latents = latents

                    # Process output
                    audio_img = sample.images[0]
                    audio = denormalize_spectrogram(audio_img)
                    audio = self.vocoder.inference(audio, lengths=160000)[0]
                    audio = audio[: int(chunk_duration_actual * 16000)]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    audio_outputs.append(audio)

                    # Process video clip
                    video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
                    video_clips.append(video_clip)

                    # Free up memory
                    del (
                        time_frames,
                        images,
                        image_embeddings,
                        time_condition,
                        sample,
                        audio_img,
                        audio,
                        video_clip,
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

                    progress_bar.update(1)
                    progress_bar.set_postfix({"Chunk": f"{chunk_idx + 1}/{num_chunks}"})

            finally:
                progress_bar.close()

                # Save processed chunks if any are available
                if len(audio_outputs) > 0 and len(video_clips) > 0:
                    # Combine audio outputs
                    full_audio = np.concatenate(audio_outputs)

                    # Set output directory
                    if overwrite_checkbox:
                        name = "output"
                        output_dir = self.savedir_sample
                    else:
                        name = datetime.now().strftime("%Y%m%d%H%M%S")
                        output_dir = os.path.join(self.savedir_sample, name)
                        os.makedirs(output_dir, exist_ok=True)

                    # Save audio
                    audio_save_path = osp.join(output_dir, "audio")
                    os.makedirs(audio_save_path, exist_ok=True)
                    save_path = osp.join(audio_save_path, f"{name}.wav")
                    sf.write(save_path, full_audio, 16000)

                    # Save video with combined audio
                    audio_clip = AudioFileClip(save_path)
                    final_video_clip = concatenate_videoclips(video_clips)
                    final_video_clip.audio = audio_clip
                    output_video_path = osp.join(output_dir, f"{name}.mp4")
                    final_video_clip.write_videofile(output_video_path)
                    save_sample_path = output_video_path

                    print(f"Saved partially processed video to {save_sample_path}")

            # Move models back to CPU
            self.time_detector.to("cpu")
            self.pipeline.to("cpu")
            self.vocoder.to("cpu")
            self.image_encoder.to("cpu")
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
                            label="Sampling steps", value=35, minimum=10, maximum=100, step=5
                        )
                    cfg_scale_slider = gr.Slider(label="CFG Scale", value=7.5, minimum=0, maximum=20)

                with gr.Accordion("Advanced Settings", open=False):
                    chunk_duration_slider = gr.Slider(
                        label="Chunk Duration (seconds)", value=2, minimum=2, maximum=60, step=2
                    )
                    use_vision_model_checkbox = gr.Checkbox(
                        label="Use Vision Model for Prompt Extraction", value=True
                    )

                with gr.Row():
                    seed_textbox = gr.Textbox(label="Seed", value=1337)
                    seed_button = gr.Button(value="\U0001f3b2", elem_classes="toolbutton")
                seed_button.click(fn=lambda x: random.randint(1, int(1e8)), outputs=[seed_textbox], queue=False)

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
                use_vision_model_checkbox,
                chunk_duration_slider,
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
                    True,
                    10,
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
                    True,
                    10,
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
                    True,
                    10,
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
                    True,
                    10,
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
                use_vision_model_checkbox,
                chunk_duration_slider,
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
