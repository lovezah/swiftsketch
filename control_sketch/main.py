import config
from PIL import Image
from my_utils import utils
from torchvision import transforms
from painter_params import Painter, PainterOptimizer
from sds_file import ControlSDSLoss
from tqdm.auto import tqdm

def load_painter(args, image, mask):
    painter = Painter(num_strokes = args.num_strokes,
                        args = args,
                        num_segments = args.num_segments,
                        device = args.device,
                        target_im = image,
                        mask = mask,
                        )
    painter = painter.to(args.device)
    return painter

def get_image_and_mask(args):
    image = Image.open(args.target) # original image
    if image.mode == "RGBA"
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, (0, 0), image)
        image = new_image
    image = image.convert("RGB")

    mask = utils.get_prob_mask(image, args.device)
    image = utils.apply_mask(image, mask)

    image = image.resize((args.render_size, args.render_size))
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]

    args.input_image = image
    args.mask = mask
    image.save(f"{args.output_dir}/input.png")
    data_transform = transforms.ToTensor()
    image = data_transform(image).unsqueeze(0).to(args.device)

    # image : (1, 3, 512, 512)
    # mask  : (512, 512)
    return image, mask

def main(args):
    print("Starting...", flush = True)

    image, mask = get_image_and_mask(args)
    painter = load_painter(args, image, mask)
    sds_loss = ControlSDSLoss(args, args.device)
    optimizer = PainterOptimizer(args, painter)

    img = painter.init_image()
    optimizer.init_optimizers()

    # optimization
    epoch_range = tqdm(range(args.num_iter + 1))
    image = image.detach()

    for epoch in epoch_range:
        optimizer.zero_grad_()
        sketches = painter.get_image().to(args.device)
        loss = sds_loss(sketches)
        loss.backward()
        optimizer.step_()

    # save
    painter.save_svg(args.output_dir, "final_svg")
    final_sketch_num = utils.read_svg(f"{args.output_dir}/final_svg.svg", args.device, multiply=True,
                                      args=None).cpu().numpy()
    final_sketch = Image.fromarray((final_sketch_num * 255).astype('uint8'), 'RGB')
    final_sketch.save(f"{args.output_dir}/final_sketch.png")
    print("Final sketch saved at:", f"{args.output_dir}/final_sketch.png", flush=True)

if __name__ == "__main__":
    args = config.parse_arguments()
    try:
        main(args)
    except Exception as e:
        print("Error:", e, flush = True)