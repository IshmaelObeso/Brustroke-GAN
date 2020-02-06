from flask import Flask, request, render_template



from utils import paint_stroke_by_stroke
from predict import *
import os
import io
import imageio
import base64
import time
import sys

# Other imports

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    template_folder = 'templates'
    static_folder = 'static'
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'GET':
        return render_template("main.html")
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        input_image = file.read()
        input_data = input_fn(image_bytes=input_image)
        model = model_fn()
        epochs = int(request.form['epochs'])
        image, base_image, conditions = predict_fn(input_data=input_data, model=model, epochs=epochs)
        # create frames for gif
        frames = paint_stroke_by_stroke(conditions, model)
        gif_images = []
        for idx, frame in enumerate(frames):
            gif_image = image
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gif_image = VF.to_tensor(gif_image).to(device)
            imagex = VF.to_pil_image(torch.cat((gif_image.clone().cpu(), VF.to_tensor(frame)), dim=2))
            gif_images.append(imagex)

        # delete old gif if any
        if os.path.exists(static_folder):
            print('path exists')
            for file in os.listdir(static_folder):
                if file.endswith('.gif'):
                    print('found a gif: {}!'.format(file))
                    print('deleting old gif')
                    os.remove(os.path.join(static_folder, file))
        print(f'static folder {static_folder}')

        gif_name = 'output_{}.gif'.format(time.time())
        print(f'gif_name {gif_name}')
        filename = static_folder + '/' + gif_name
        print(f'file_name {filename}')
        imageio.mimsave(filename, gif_images, fps=3)

        # create file-object in memory
        img_io = io.BytesIO()
        base_image_io = io.BytesIO()
        # write PNG in file-object
        image.save(img_io, 'PNG')
        base_image.save(base_image_io, 'PNG')

        # move to beginning of file
        img_io.seek(0, 0)
        base_image_io.seek(0, 0)

        # encode file in b64
        image = base64.b64encode(img_io.getvalue())
        base_image = base64.b64encode(base_image_io.getvalue())
        # close file object
        img_io.close()
        base_image_io.close()
        # decode
        image = image.decode('ascii')
        base_image = base_image.decode('ascii')
        # send to template
        return render_template('result.html', original_image=base_image, output_image=image, filename=gif_name)


if __name__ == "__main__":
    app.run()


