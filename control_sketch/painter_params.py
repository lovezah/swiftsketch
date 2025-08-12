import torch
import torch.nn.functional as F
import pydiffvg
import random


class Painter(torch.nn.Module):
    def __init__(self, args, )


    def init_image(self):
        for i in range(self.num_paths):
            stroke_color = torch.tensor([0., 0., 0., 1.]) # RGBA : black
            path = self.get_path()
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]), stroke_color = stroke_color, fill_color = None)
            self.shape_groups.append(path_group)
        self.optimize_flag = [True for i in range(len(self.shapes))]



    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)

        p0 = self.inds_normalised[self.strokes_counter] if self.use_init_method else (random.random(), random.random())
        self.initial_points.append(p0)
        points.append(p0) 
        for j in range(self.num_segments):  # here is 1 by defult
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                self.initial_points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)  

        self.strokes_counter += 1
        return path


    def get_image(self):
        # convert RGBA to RGB
        image = self.render_warp()
        f = image[:, :, 3:]
        image = f * image[:, :, :3] + (1 - f) * torch.ones(image.shape[0], image.shape[1], 3, device = self.device)
        image = image[:, :, :3].unsqueeze(0) # (1, H, W, 3)
        image = image.permute(0, 3, 1, 2).to(self.device) # (1, 3, H, W)
        return image

    def render_warp(self):  
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width,  
                      self.canvas_height, 
                      2,  
                      2,  
                      0,  
                      None,
                      *scene_args)
        # （R, G, B, A）
        # shape = (H, W, 4)
        return img

    
    def set_attention_map(self):
        if hasattr(self.args, 'attn_from_dict'): 
            attn = self.args.attn_from_dict
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
            if hasattr(self.args, 'obj_bb'): 
                attn = np.stack([attn] * 3, axis=-1)
                x0, x1, y0, y1= self.args.obj_bb
                attn = utils.cut_and_resize(attn, x0, x1, y0, y1, self.args.new_height, self.args.new_width, "mask")
                attn = torch.from_numpy(attn[:, :, 0])
        elif self.attn_model == "clip" or self.args.object_name == "" :
            self.saliency_clip_model = "ViT-B/32"
            self.define_clip_attention_input(self.target_im)
            attn= self.clip_attn()
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
        else: # self.attn_model == "diffusion":
            attn= self.diffusion_attn()
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
        return attn

    
    def set_attention_threshold_map(self):
        attn_map= torch.pow(self.attention_map, 2)
        attn_map_to_plot = (attn_map * self.mask) 
        weights = attn_map.numpy().astype(np.float32)
        
        mask= self.mask
        mask = (mask / mask.max()) * 255
        mask = mask.numpy().astype(np.uint8)

        self.inds, self.clustered_mask_to_plot = self.get_points_smart_clustering(mask, weights)

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 0] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 1] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        return attn_map_to_plot
