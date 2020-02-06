import PIL
import torch
import torchvision.transforms.functional as VF

def load_image(input_data, size=None, scale=None):
  # Load an image from a given file name
    img = input_data.convert("RGB")
    if size is not None:
        img.thumbnail((size, size), PIL.Image.ANTIALIAS)
    elif scale is not None:
        size = img.size[0] * scale
        img.thumbnail((size, size), PIL.Image.ANTIALIAS)
    return img

def paint_canvas(generated_strokes, size, darkness=1):
    canvas = torch.ones_like(generated_strokes[0], requires_grad=True, dtype=torch.float32)
    
    for idx, stroke in enumerate(generated_strokes):
        # extract the strokes's darkest dominant (rgb) color
        brush_color = stroke.min(1)[0].min(1)[0].clamp_(0,1).view(-1, 3, 1, 1)
        brush_color = brush_color.expand(-1, 3, size, size)
        
        # get the 'darkness' of each individual pixel in a stroke by averaging
        darkness_mask = torch.mean(stroke, 0)
        
        # make the value of a darker stroke higher
        darkness_mask = 1 - torch.reshape(darkness_mask, (-1, 1, size, size))
        
        # Scale this darkness mask from 0 to 1
        darkness_mask = darkness_mask / torch.max(darkness_mask)
        
        # Replace the original stroke with one that has all colored pixels set to the
        # actual color used
        stroke_whitespace = torch.eq(stroke, 1.)
        
        maxed_stroke = torch.where(stroke_whitespace, stroke, brush_color)
        
        # Linearly blend
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        maxed_stroke = maxed_stroke.to(device)
        darkness_mask = (darkness * darkness_mask).to(device)
        canvas = (darkness_mask) * maxed_stroke + (1-darkness_mask) * canvas
    return canvas


def paintx(generated_strokes):
    canvas = paint_canvas(generated_strokes, size = generated_strokes[0].shape[-1]).squeeze()
    
    return canvas.view(1, *canvas.shape)
        
def paint_stroke_by_stroke(condition, generator, size=64, darkness=1):
    frames = []
    
    generated_strokes = generator(condition)
    
    canvas = torch.ones_like(generated_strokes[0], requires_grad=True, dtype=torch.float32)
    
    for stroke in generated_strokes:
        # extract the stroke's darkest dominant (rgb) color
        brush_color = stroke.min(1)[0].min(1)[0].clamp_(0, 1).view(-1, 3, 1, 1)
        brush_color = brush_color.expand(-1, 3, size, size)

        # Get the "darkness" of each individual pixel in a stroke by averaging.
        darkness_mask = torch.mean(stroke, 0)

        # Make the value of a darker stroke higher.
        darkness_mask = 1 - torch.reshape(darkness_mask, (-1, 1, size, size))

        # Scale this darkness mask from 0 to 1.
        darkness_mask = darkness_mask / torch.max(darkness_mask)

        # Replace the original stroke with one that has all colored pixels set to the
        # actual color used.
        stroke_whitespace = torch.eq(stroke, 1.)

        #maxed_stroke = torch.where(stroke_whitespace, stroke, brush_color)
        maxed_stroke = torch.where(stroke_whitespace, stroke, brush_color)

        # Linearly blend
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        maxed_stroke = maxed_stroke.to(device)
        darkness_mask = (darkness * darkness_mask).to(device)
        canvas = (darkness_mask) * maxed_stroke + (1-darkness_mask) * canvas
        
        img_result = VF.to_pil_image(canvas.clone().cpu().squeeze())
        frames.append(img_result)

    return frames