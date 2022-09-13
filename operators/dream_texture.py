import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty
import asyncio
import os
from ..async_loop import *
from ..pil_to_image import *
from ..prompt_engineering import *
from ..absolute_path import WEIGHTS_PATH, absolute_path
from .install_dependencies import are_dependencies_installed

sampler_options = [
    ("ddim", "DDIM", "", 1),
    ("plms", "PLMS", "", 2),
    ("k_lms", "KLMS", "", 3),
    ("k_dpm_2", "KDPM_2", "", 4),
    ("k_dpm_2_a", "KDPM_2A", "", 5),
    ("k_euler", "KEULER", "", 6),
    ("k_euler_a", "KEULER_A", "", 7),
    ("k_heun", "KHEUN", "", 8),
]

# A shared `Generate` instance.
# This allows the slow model loading process to happen once,
# and re-use the model on subsequent calls.
generator = None

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    # Prompt
    prompt_structure: EnumProperty(name="Preset", items=prompt_structures_items, description="Fill in a few simple options to create interesting images quickly")

    # Size
    width: IntProperty(name="Width", default=512)
    height: IntProperty(name="Height", default=512)

    # Simple Options
    seamless: BoolProperty(name="Seamless", default=False, description="Enables seamless/tilable image generation")

    # Advanced
    show_advanced: BoolProperty(name="", default=False)
    seed: IntProperty(name="Seed", default=-1, description="Seed for RNG. Using the same seed will give the same image. A seed of '-1' will pick a random seed each time")
    full_precision: BoolProperty(name="Full Precision", default=False, description="Whether to use full precision or half precision floats. Full precision is slower, but required by some GPUs")
    iterations: IntProperty(name="Iterations", default=1, min=1, description="How many images to generate")
    steps: IntProperty(name="Steps", default=25, min=1)
    cfgscale: FloatProperty(name="CFG Scale", default=7.5, min=1, description="How strongly the prompt influences the image")
    sampler: EnumProperty(name="Sampler", items=sampler_options, default=3)
    show_steps: BoolProperty(name="Show Steps", description="Displays intermediate steps in the Image Viewer. Disabling can speed up generation", default=True)

    # Init Image
    use_init_img: BoolProperty(name="", default=False)
    strength: FloatProperty(name="Strength", default=0.75, min=0, max=1)
    fit: BoolProperty(name="Fit to width/height", default=True)

    @classmethod
    def poll(self, context):
        return True
    
    def invoke(self, context, event):
        weights_installed = os.path.exists(WEIGHTS_PATH)
        if not weights_installed or not are_dependencies_installed():
            self.report({'ERROR'}, "Please complete setup in the preferences window.")
            return {"FINISHED"}
        else:
            return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        
        prompt_box = layout.box()
        prompt_box_heading = prompt_box.row()
        prompt_box_heading.label(text="Prompt")
        prompt_box_heading.prop(self, "prompt_structure")
        structure = next(x for x in prompt_structures if x.id == self.prompt_structure)
        for segment in structure.structure:
            segment_row = prompt_box.row()
            enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
            is_custom = getattr(context.scene, enum_prop) == 'custom'
            if is_custom:
                segment_row.prop(context.scene, 'prompt_structure_token_' + segment.id)
            segment_row.prop(context.scene, enum_prop, icon_only=is_custom)
        
        size_box = layout.box()
        size_box.label(text="Configuration")
        size_box.prop(self, "width")
        size_box.prop(self, "height")
        size_box.prop(self, "seamless")
        
        init_img_box = layout.box()
        init_img_heading = init_img_box.row()
        init_img_heading.prop(self, "use_init_img")
        init_img_heading.label(text="Init Image")
        if self.use_init_img:
            init_img_box.template_ID(context.scene, "init_img", open="image.open")
            init_img_box.prop(self, "strength")
            init_img_box.prop(self, "fit")

        advanced_box = layout.box()
        advanced_box_heading = advanced_box.row()
        advanced_box_heading.prop(self, "show_advanced", icon="DOWNARROW_HLT" if self.show_advanced else "RIGHTARROW_THIN", emboss=False, icon_only=True)
        advanced_box_heading.label(text="Advanced Configuration")
        if self.show_advanced:
            advanced_box.prop(self, "full_precision")
            advanced_box.prop(self, "seed")
            # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
            advanced_box.prop(self, "steps")
            advanced_box.prop(self, "cfgscale")
            advanced_box.prop(self, "sampler")
            advanced_box.prop(self, "show_steps")

    def cancel(self, context):
        pass

    async def dream_texture(self, context):
        structure = next(x for x in prompt_structures if x.id == self.prompt_structure)
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__
        tokens = {}
        for segment in structure.structure:
            enum_value = getattr(context.scene, 'prompt_structure_token_' + segment.id + '_enum')
            if enum_value == 'custom':
                tokens[segment.id] = getattr(context.scene, 'prompt_structure_token_' + segment.id)
            else:
                tokens[segment.id] = next(x for x in segment.values if x[0] == enum_value)[1]
        generated_prompt = structure.generate(dotdict(tokens))

        # Support Apple Silicon GPUs as much as possible.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        from ..stable_diffusion.ldm.generate import Generate
        from omegaconf import OmegaConf
        
        config  = absolute_path('stable_diffusion/configs/models.yaml')
        model   = 'stable-diffusion-1.4'

        models  = OmegaConf.load(config)
        width   = models[model].width
        height  = models[model].height
        config  = absolute_path('stable_diffusion/' + models[model].config)
        weights = absolute_path('stable_diffusion/' + models[model].weights)

        global generator
        if generator is None:
            generator = Generate(
                width=width,
                height=height,
                sampler_name=self.sampler,
                weights=weights,
                full_precision=self.full_precision,
                seamless=self.seamless,
                config=config,
            )
            generator.load_model()

        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        screen = context.screen
        last_data_block = None
        scene = context.scene
        def image_writer(image, seed, upscaled=False):
            nonlocal last_data_block
            # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
            if not upscaled:
                if last_data_block is not None:
                    bpy.data.images.remove(last_data_block)
                    last_data_block = None
                image = pil_to_image(image, name=f"{seed}")
                if node_tree is not None:
                    nodes = node_tree.nodes
                    texture_node = nodes.new("ShaderNodeTexImage")
                    texture_node.image = image
                    nodes.active = texture_node
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.spaces.active.image = image
        def view_step(samples, step):
            nonlocal last_data_block
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    step_image = pil_to_image(generator._sample_to_image(samples), name=f'Step {step + 1}/{self.steps}')
                    area.spaces.active.image = step_image
                    if last_data_block is not None:
                        bpy.data.images.remove(last_data_block)
                    last_data_block = step_image
                    return # Only perform this on the first image editor found.

        report = self.report
        def log_step(samples, step):
            report({'ERROR'}, f"Step {step + 1}/{self.steps}")
            # def step_menu(self, context):
            #     self.layout.label(text=f"Step {step + 1}/{self.steps}")
            # window_manager.popup .popup_menu(step_menu, title = "Dream Texture Progress", icon = "INFO")

        def perform():
            generator.prompt2image(
                # prompt string (no default)
                prompt=generated_prompt,
                # iterations (1); image count=iterations
                iterations=self.iterations,
                # refinement steps per iteration
                steps=self.steps,
                # seed for random number generator
                seed=None if self.seed == -1 else self.seed,
                # width of image, in multiples of 64 (512)
                width=self.width,
                # height of image, in multiples of 64 (512)
                height=self.height,
                # how strongly the prompt influences the image (7.5) (must be >1)
                cfg_scale=self.cfgscale,
                # path to an initial image - its dimensions override width and height
                init_img=scene.init_img.filepath if scene.init_img is not None and self.use_init_img else None,

                fit=self.fit,
                # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
                strength=self.strength,
                # strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
                gfpgan_strength=0.0, # 0 disables upscaling, which is currently not supported by the addon.
                # image randomness (eta=0.0 means the same seed always produces the same image)
                ddim_eta=0.0,
                # a function or method that will be called each step
                step_callback=view_step if self.show_steps else log_step,
                # a function or method that will be called each time an image is generated
                image_callback=image_writer,
                
                sampler_name=self.sampler
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, perform)

    def execute(self, context):
        async_task = asyncio.ensure_future(self.dream_texture(context))
        # async_task.add_done_callback(done_callback)
        ensure_async_loop()

        return {'FINISHED'}