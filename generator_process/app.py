from actions.prompt_to_image import Pipeline, Scheduler, Optimizations, StepPreviewMode, approximate_decoded_latents, ImageGenerationResult

from PIL import Image
import numpy as np
import bpy

from actor import Actor

from flask import Flask, Response, request


app = Flask(__name__)


class Generator(Actor):
	from actions.depth_to_image import depth_to_image


@app.route('/depth/predict', methods=['OPTIONS', 'POST'])
def predict():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	strength = 0.1
	steps = 1
	seed = 1
	cfg_scale = 0.5
	use_negative_prompt = False
	negative_prompt = 'test'
	step_preview_mode = 'Fast'
	# **kwargs

	prompt = request.args.get('prompt')

	COLOR_IMAGE = request.files['color']
	DEPTH_IMAGE = request.files['depth']

	image = np.asarray(Image.open(COLOR_IMAGE) if COLOR_IMAGE is not None else None)
	depth = np.asarray(Image.open(DEPTH_IMAGE))

	texture = None

	def on_response(_, response):
		nonlocal texture
		if texture is None:
			texture = bpy.data.images.new(name="Step", width=response.image.shape[1], height=response.image.shape[0])
		texture.name = f"Step {response.step}/{context.scene.dream_textures_project_prompt.steps}"
		texture.pixels[:] = response.image.ravel()
		texture.update()
		image_texture_node.image = texture

	def on_done(future):
		nonlocal texture
		generated = future.result()
		if isinstance(generated, list):
			generated = generated[-1]
		if texture is None:
			texture = bpy.data.images.new(name=str(generated.seed), width=generated.image.shape[1], height=generated.image.shape[0])
		texture.name = str(generated.seed)
		material.name = str(generated.seed)
		texture.pixels[:] = generated.image.ravel()
		texture.update()
		image_texture_node.image = texture

	future = Generator.shared().depth_to_image(
		pipeline=Pipeline.STABLE_DIFFUSION,
		model='',
		scheduler=Scheduler,
		optimizations=Optimizations(),
		depth=depth,
		image=image,
		strength=0.1,
		prompt=prompt,
		steps=1,
		seed=1,
		cfg_scale=0.5,
		use_negative_prompt=False,
		negative_prompt='test',
		step_preview_mode='Fast',
	)

	future.add_response_callback(on_response)
	future.add_done_callback(on_done)

	# print(type(texture))

	texture.save('test.png')

	img_byte_arr = io.BytesIO()
	output.images[0].save(img_byte_arr, format='PNG')
	img_byte_arr = img_byte_arr.getvalue()
	response = Response(img_byte_arr, headers={'Content-Type':'image/png'})
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Headers'] = '*'
	response.headers['Access-Control-Allow-Methods'] = '*'
	response.headers['Access-Control-Expose-Headers'] = '*'
	response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
	response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
	response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
	return response


if __name__ == '__main__':
	CONVERTED_MODEL_PATH = 'carsonkatri/stable-diffusion-2-depth-diffusers'

	app.run(host='0.0.0.0', port=8081, threaded=True, debug=False)
