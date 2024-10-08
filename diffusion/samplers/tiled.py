from dataclasses import dataclass, field
from functools import cached_property, lru_cache
import random
import time
from typing import List
import torch
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
from diffusion.samplers.sampler import Sampler
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
import matplotlib.pyplot as plt

from dataclasses import dataclass
import networkx as nx

@dataclass
class Tile:
    """A tile of an image."""
    image: torch.Tensor
    model_output: torch.Tensor = None  # The model output for the next level
    prev_model_output: torch.Tensor = None  # To merge with model outputs that are one level lower
    sampler_prev_model_output: torch.Tensor = None  # For higher order sampling
    level: int = 0

class TiledSampler(Sampler):
    def __init__(self, model, scheduler, overlap=16, timesteps=15, seed=None, device='cpu',
                 parent_sampler=None, boundary=None, batch_size=1, generation_batch_size=1):
        """Initialize a TiledSampler for efficient large image generation.

        This sampler divides the input image into tiles, processes each tile
        independently, and then combines the results. This approach allows for
        generating large images that wouldn't fit in memory if processed all at once.

        Args:
            model (torch.nn.Module): The diffusion model to use for sampling.
            scheduler (torch.nn.Module): The noise scheduler for the diffusion process.
            overlap (int, optional): The overlap between sampling tiles. Defaults to 16. MUST divide the image size evenly.
            scale_factor (int, optional): The scale factor to use for the parent (conditional) image. Defaults to 4.
                i.e scale_factor^2 pixels in this image are represented by 1 pixel in the parent image. Unused if parent_sampler is None.
            parent_sampler (Sampler, optional): The sampler to use for the parent (conditional) image. Defaults to None.
            boundary (tuple[int, int, int, int], optional): The boundary of the region to sample. Defaults to None.
            batch_size (int, optional): The batch size of tiles, i.e how many tiles are stacked together.
            generation_batch_size (int, optional): Maximum number of tiles to process at once. The effective batch size of model inputs is batch_size * generation_batch_size.
        """
        self.model = model
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.overlap = overlap
        self._tile_size = self.model.config.image_size
        self._seed = seed if seed is not None else random.randint(0, 2**30)
        self.device = device
        self.tiles = {}
        self.parent_sampler = parent_sampler
        self.boundary = boundary
        self.batch_size = batch_size
        self.generation_batch_size = generation_batch_size
        
    @property
    def seed(self):
        return self._seed
    
    @property
    def tile_size(self):
        return self._tile_size
        
    @cached_property
    def weights(self):
        s = self.tile_size
        mid = (s - 1) / 2
        y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
        epsilon = 1e-3
        distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
        distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
        return distance_y * distance_x
    
    @cached_property
    def tile_boundary(self):    
        if self.boundary is None:
            return None
        
        tile_boundary_left = (self.boundary[0]) // (self.tile_size - self.overlap)
        tile_boundary_right = (self.boundary[2] - 1) // (self.tile_size - self.overlap)
        tile_boundary_top = (self.boundary[1]) // (self.tile_size - self.overlap)
        tile_boundary_bottom = (self.boundary[3] - 1) // (self.tile_size - self.overlap)
        return tile_boundary_left, tile_boundary_top, tile_boundary_right, tile_boundary_bottom
        
    def is_tile_in_boundary(self, tile_y, tile_x): 
        if self.tile_boundary is None:
            return True
        
        tile_boundary_left, tile_boundary_top, tile_boundary_right, tile_boundary_bottom = self.tile_boundary
        return (tile_boundary_left <= tile_x < tile_boundary_right and
                tile_boundary_top <= tile_y < tile_boundary_bottom)
        
    @lru_cache(maxsize=100)  # Cache the results because this is called a lot and often with the same arguments
    def coord_to_seed(self, tile_y, tile_x):
        # Convert to 32-bit unsigned integers, handling negative inputs
        tile_y = (tile_y & 0xFFFFFFFF)
        tile_x = (tile_x & 0xFFFFFFFF)
        seed = self.seed & 0xFFFFFFFF

        # Combine tile_y, tile_x, and seed using FNV-1a hash
        hash_value = 2166136261  # FNV offset basis for 32-bit

        # Hash tile_y
        hash_value ^= tile_y
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF  # FNV prime for 32-bit

        # Hash tile_x
        hash_value ^= tile_x
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF

        # Hash seed
        hash_value ^= seed
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF

        return hash_value

    def get_tile(self, tile_y, tile_x):
        coords = (tile_y, tile_x)
        tile = self.tiles.get(coords)
        if tile is None:
            if not self.is_tile_in_boundary(tile_y, tile_x):
                return None
            
            image = torch.zeros(self.batch_size, self.model.config.out_channels, self.tile_size, self.tile_size)
            
            top = tile_y * (self.tile_size - self.overlap)
            left = tile_x * (self.tile_size - self.overlap)
            center = torch.randn(self.batch_size, self.model.config.out_channels, self.tile_size-self.overlap*2, self.tile_size-self.overlap*2, 
                                 generator=torch.Generator().manual_seed(self.coord_to_seed(top + self.overlap, left + self.overlap)))
            image[..., self.overlap:self.tile_size-self.overlap, self.overlap:self.tile_size-self.overlap] = center
            
            # Do all edges one at a time to ensure consistency
            if self.overlap > 0:
                for i in range(self.tile_size // self.overlap):
                    for j in range(self.tile_size // self.overlap):
                        # Ignore center tiles
                        if (0 < i < self.tile_size // self.overlap - 1) and (0 < j < self.tile_size // self.overlap - 1):
                            continue
                        patch_left = left + self.overlap * j
                        patch_top = top + self.overlap * i
                        patch_local_left = self.overlap * j
                        patch_local_top = self.overlap * i
                        patch = torch.randn(self.batch_size, self.model.config.out_channels, self.overlap, self.overlap, 
                                            generator=torch.Generator().manual_seed(self.coord_to_seed(patch_top, patch_left)))
                        image[..., patch_local_top:patch_local_top+self.overlap, patch_local_left:patch_local_left+self.overlap] = patch
                
            tile = Tile(image * self.scheduler.sigmas[0])
            self.tiles[coords] = tile
        return tile
    
    def get_merged_tile(self, tile_y, tile_x):
        """Merge model outputs from a tile and its neighbors."""
        base_tile = self.get_tile(tile_y, tile_x)
        assert base_tile is not None, "Tile is out of boundary"
        
        weights = torch.zeros_like(base_tile.image)
        merged_model_output = torch.zeros_like(base_tile.image)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if self.overlap == 0 and (i != 0 or j != 0):
                    continue
                    
                tile = self.get_tile(tile_y + i, tile_x + j)
                if tile is None:
                    continue
                assert tile.level == base_tile.level or base_tile.level == tile.level - 1

                # i = -1 -> top = 48, i = 0 -> top = 0, i = 1 -> top = 0
                top = max(-i, 0) * (self.tile_size - self.overlap)
                # i = -1 -> bottom = 64, i = 0 -> bottom = 64, i = 1 -> bottom = 16
                bottom = self.tile_size - max(i, 0) * (self.tile_size - self.overlap)

                # j = -1 -> left = 48, j = 0 -> left = 0, j = 1 -> left = 0
                left = max(-j, 0) * (self.tile_size - self.overlap)
                # j = -1 -> right = 64, j = 0 -> right = 64, j = 1 -> right = 16
                right = self.tile_size - max(j, 0) * (self.tile_size - self.overlap)

                refl_top = self.tile_size - bottom
                refl_bottom = self.tile_size - top
                refl_left = self.tile_size - right
                refl_right = self.tile_size - left

                if tile.level == base_tile.level:
                    other_output = tile.model_output
                else:
                    # When base_tile is one level lower, the model output is stored in prev_model_output
                    other_output = tile.prev_model_output

                other_output_region = other_output[..., top:bottom, left:right]
                weights_region = self.weights[None, None, top:bottom, left:right].expand(self.batch_size, -1, -1, -1)

                merged_model_output[..., refl_top:refl_bottom, refl_left:refl_right] += other_output_region * weights_region
                weights[..., refl_top:refl_bottom, refl_left:refl_right] += weights_region

        assert (weights > 0).all().item(), "Weight is zero"
        return merged_model_output / weights
        
    def scheduler_step(self, tiles_y, tiles_x):
        """Upgrade a tile to the next level. For each tile, the tile and surrounding tiles must have a model output at the same level.
        This method takes a list of tiles and upgrades them in batches, separated by the level of the tiles.
        """
        assert len(tiles_x) == len(tiles_y)
        if len(tiles_y) == 0:
            return
        
        input_samples = []
        input_preds = []
        t = torch.zeros(len(tiles_y), dtype=torch.float32)
        sigmas = torch.zeros(len(tiles_y), dtype=torch.float32)
        for i, (tile_y, tile_x) in enumerate(zip(tiles_y, tiles_x)):
            base_tile = self.get_tile(tile_y, tile_x)
            assert base_tile is not None, "Tile is out of boundary"
            t[i] = self.scheduler.timesteps[base_tile.level]
            sigmas[i] = self.scheduler.sigmas[base_tile.level]
            input_samples.append(base_tile.image)
            input_preds.append(self.get_merged_tile(tile_y, tile_x))
            
        input_sample = torch.stack(input_samples)
        pred_noise = torch.stack(input_preds)
        t = t.to(self.device)
        
        unique_t = torch.unique(t)
        prev_samples = torch.zeros_like(input_sample)
        for unique_t_value in unique_t:
            mask = t == unique_t_value
            indices = torch.argwhere(mask).flatten().numpy()
            
            batch_pred_noise = pred_noise[mask]
            batch_input_sample = input_sample[mask]
            
            # Reset the step index to None because the scheduler is being used by many different tiles
            self.scheduler._begin_index = None
            self.scheduler._init_step_index(unique_t_value)  # This is evil but it works
            self.scheduler.lower_order_nums = min(self.scheduler.config.solver_order, self.scheduler._step_index)
            self.scheduler.model_outputs = [None] * self.scheduler.config.solver_order
            
            # Set scheduler model outputs so that gradients can be computed. Not applicable for the first timestep or if the solver order is 1.
            if self.scheduler.config.solver_order != 1 and unique_t_value != self.scheduler.timesteps[0]:
                sampler_prev_model_output = torch.zeros_like(batch_pred_noise)
                for i in range(len(sampler_prev_model_output)):
                    tile = self.get_tile(tiles_y[indices[i]], tiles_x[indices[i]])
                    sampler_prev_model_output[i] = tile.sampler_prev_model_output
                self.scheduler.model_outputs[-1] = sampler_prev_model_output.view(-1, *batch_pred_noise.shape[2:])
            
            # Reshape inputs to account for batch dimension
            reshaped_pred_noise = batch_pred_noise.view(-1, *batch_pred_noise.shape[2:])
            reshaped_input_sample = batch_input_sample.view(-1, *batch_input_sample.shape[2:])
            
            batch_prev_sample = self.scheduler.step(reshaped_pred_noise, unique_t_value, reshaped_input_sample).prev_sample
            
            # Reshape output back to original shape
            batch_prev_sample = batch_prev_sample.view(*batch_input_sample.shape)
            prev_samples[mask] = batch_prev_sample
            
            # Store the sampler_prev_model_output for each tile
            if self.scheduler.config.solver_order != 1:
                for i in range(len(batch_prev_sample)):
                    model_output = self.scheduler.model_outputs[-1].view(*batch_input_sample.shape)[i]
                    tile = self.get_tile(tiles_y[indices[i]], tiles_x[indices[i]])
                    tile.sampler_prev_model_output = model_output
        
        # Update level of each tile
        for i, (tile_y, tile_x) in enumerate(zip(tiles_y, tiles_x)):
            tile = self.get_tile(tile_y, tile_x)
            tile.image = prev_samples[i]
            tile.prev_model_output = tile.model_output
            tile.model_output = None
            tile.level += 1

    def create_model_output(self, tiles_y, tiles_x, **net_inputs):
        """Get the model output for many tiles, in a batch."""
        assert len(tiles_x) == len(tiles_y)
        if len(tiles_y) == 0:
            return
        
        input_samples = []
        t = torch.zeros(len(tiles_y), dtype=torch.float32)
        sigmas = torch.zeros(len(tiles_y), dtype=torch.float32)
        for i, (tile_y, tile_x) in enumerate(zip(tiles_y, tiles_x)):
            base_tile = self.get_tile(tile_y, tile_x)
            assert base_tile is not None, "Tile is out of boundary"
            t[i] = self.scheduler.timesteps[base_tile.level]
            sigmas[i] = self.scheduler.sigmas[base_tile.level]
            input_samples.append(base_tile.image)
            
        input_sample = torch.cat(input_samples, dim=0)
        t = t.to(self.device).repeat(self.batch_size)
        sigmas = sigmas.repeat(self.batch_size)
        x = self.scheduler.precondition_inputs(input_sample, sigmas.view(-1, 1, 1, 1))
        model_outputs = self.model(x, t, **net_inputs)
        for i in range(len(tiles_y)):
            tile = self.get_tile(tiles_y[i], tiles_x[i])
            tile.model_output = model_outputs[i*self.batch_size:(i+1)*self.batch_size]

    def upgrade_tiles(self, tile_y, tile_x, target_levels, use_tqdm=True):
        """Upgrades tiles to a higher level.
        
        Args:
            tile_y (list[int]): The y coordinates of the tiles.
            tile_x (list[int]): The x coordinates of the tiles.
            target_levels (list[int]): The levels to upgrade the tiles to.
        """
        self.scheduler.set_timesteps(self.timesteps)
    
        @dataclass
        class TileNode:
            tile_y: int
            tile_x: int
            level: int
            dependents: list = field(default_factory=list)  # Tiles that depend on this one being upgraded
            dependencies: list = field(default_factory=list)  # Tiles that must be upgraded before this one
        
        nodes = {(a, b, c): TileNode(a, b, c) for a, b, c in zip(tile_y, tile_x, target_levels)}
        
        # Filter out nodes that are already at target level or not in boundary
        nodes = {
            (ty, tx, level): node
            for (ty, tx, level), node in nodes.items()
            if self.get_tile(ty, tx) is not None and self.get_tile(ty, tx).level < level
        }
        if len(nodes) == 0:
            return
        
        S = set(nodes.keys())
        while S:
            tile_y, tile_x, level = S.pop()
            node = nodes[(tile_y, tile_x, level)]
            assert self.get_tile(tile_y, tile_x) is not None, "Tile is out of boundary"
            if level == 0:
                continue
            
            for m in (-1, 0, 1):
                for n in (-1, 0, 1):
                    if self.overlap == 0 and (m != 0 or n != 0):
                        continue
                    
                    child_tile_y = tile_y + m
                    child_tile_x = tile_x + n
                    tile = self.get_tile(child_tile_y, child_tile_x)
                    if tile is None:
                        continue
                    if tile.level < level - 1 or (tile.level == level - 1 and tile.model_output is None):
                        key = (child_tile_y, child_tile_x, level - 1)
                        if key not in nodes:
                            child_node = TileNode(child_tile_y, child_tile_x, level - 1)
                            S.add(key)
                            nodes[key] = child_node
                        else:
                            child_node = nodes[key]
                        child_node.dependents.append(node)
                        node.dependencies.append(child_node)
        
        def visualize_graph():
            # Create a directed graph
            G = nx.DiGraph()
            for node in nodes.values():
                G.add_node(f"({node.tile_y},{node.tile_x},{node.level})")
                for dependent in node.dependents:
                    G.add_edge(f"({node.tile_y},{node.tile_x},{node.level})",
                               f"({dependent.tile_y},{dependent.tile_x},{dependent.level})")
    
            # Plot the graph with topological sorting
            plt.figure(figsize=(12, 8))
    
            # Use topological_generations to get layers
            generations = list(nx.topological_generations(G))
            layer_sizes = [len(gen) for gen in generations]
            max_layer_size = max(layer_sizes)
    
            # Calculate positions
            pos = {}
            y_offset = 0
            for i, gen in enumerate(generations):
                x_offset = (max_layer_size - len(gen)) / 2
                for j, node in enumerate(gen):
                    pos[node] = (x_offset + j, -i)
                y_offset += 1
    
            # Draw the graph
            nx.draw(G, pos, node_color='lightblue', 
                    node_size=2000, font_size=8, arrows=True, with_labels=False)
            nx.draw_networkx_labels(G, pos)
    
            plt.title("Topologically Sorted Tile Dependency Tree")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Process nodes in topological order
        queue = deque(node for node in nodes.values() if not node.dependencies)
        pbar = tqdm(total=len(nodes), desc="Upgrading tiles", disable=not use_tqdm)
        while queue:
            batch = []
            for _ in range(min(self.generation_batch_size, len(queue))):
                if len(queue[0].dependencies) == 0:
                    batch.append(queue.popleft())
                else:
                    break
            
            assert len(batch) > 0
            
            # To upgrade these tiles to the next level, we first need to create the model outputs for the next level, and then perform the scheduler step.
            # Filter out tiles that are already at the target level
            tiles_to_upgrade = [(n.tile_y, n.tile_x) for n in batch if self.get_tile(n.tile_y, n.tile_x).level < n.level]
            if tiles_to_upgrade:
                upgrade_y, upgrade_x = zip(*tiles_to_upgrade)
                self.scheduler_step(upgrade_y, upgrade_x)
            
            # Only create model outputs for tiles that have dependents.
            tiles_to_evaluate = [(n.tile_y, n.tile_x) for n in batch if n.dependents]
            if tiles_to_evaluate:
                create_y, create_x = zip(*tiles_to_evaluate)
                self.create_model_output(create_y, create_x)
                
            for current_node in batch:
                for dependent in current_node.dependents:
                    dependent.dependencies.remove(current_node)
                    if not dependent.dependencies:
                        queue.append(dependent)
                        
            pbar.update(len(batch))
    
    def crop_region(self, tile_y, tile_x, top, left, bottom, right):
        """Returns a region that is entirely within the tile."""
        tile_top = tile_y * (self.tile_size - self.overlap)
        tile_left = tile_x * (self.tile_size - self.overlap)
        tile_bottom = tile_top + self.tile_size
        tile_right = tile_left + self.tile_size
        return (max(top, tile_top), max(left, tile_left), 
                min(bottom, tile_bottom), min(right, tile_right))               
        
    def get_tile_bounds(self, tile_y, tile_x):
        """Returns the bounds of a tile in the coordinate system of the image.
        Format is (top, left, bottom, right)."""
        return (
            tile_y * (self.tile_size - self.overlap),
            tile_x * (self.tile_size - self.overlap),
            tile_y * (self.tile_size - self.overlap) + self.tile_size,
            tile_x * (self.tile_size - self.overlap) + self.tile_size,
        )
        
    def get_tiles_in_region(self, top, left, bottom, right):
        """Returns a list of tiles that are in the region."""
        tile_coord_left = (left - self.overlap) // (self.tile_size - self.overlap)
        tile_coord_right = (right + self.tile_size - self.overlap - 1) // (self.tile_size - self.overlap)
        tile_coord_top = (top - self.overlap) // (self.tile_size - self.overlap)
        tile_coord_bottom = (bottom + self.tile_size - self.overlap - 1) // (self.tile_size - self.overlap)
        
        return ((i, j) for i in range(tile_coord_top, tile_coord_bottom) for j in range(tile_coord_left, tile_coord_right)
                if self.is_tile_in_boundary(i, j))
    
    def get_region(self, top, left, bottom, right, generate=True):
        """Get a region of the image.

        This method retrieves a specified rectangular region from the generated image.
        It handles the complexities of tiled generation, including overlaps and boundaries.

        Args:
            top (int): The top coordinate of the region to retrieve.
            left (int): The left coordinate of the region to retrieve.
            bottom (int): The bottom coordinate of the region to retrieve.
            right (int): The right coordinate of the region to retrieve.
            upgrade_batch_size (int, optional): Batch size for upgrading tiles if needed.
                If provided, tiles in the region will be upgraded to the highest level
                before retrieval.

        Returns:
            torch.Tensor: A tensor containing the requested region of the image.
                The shape is (batch_size, channels, height, width), where height
                and width correspond to the dimensions of the requested region.
                The image will be black if the region is not fully covered by tiles.
        """
        assert top < bottom, "Top must be less than bottom"
        assert left < right, "Left must be less than right"
        
        output = torch.zeros(self.batch_size, self.model.config.out_channels, bottom-top, right-left)
        weights = torch.zeros_like(output)
        
        region_tiles = self.get_tiles_in_region(top, left, bottom, right)
        if generate:
            region_tiles = list(region_tiles)
            self.upgrade_tiles(*zip(*region_tiles), [self.timesteps] * len(region_tiles))
        for i, j in region_tiles:
            tile = self.get_tile(i, j)
            if tile is None:
                    continue
                
            tile_top, tile_left, tile_bottom, tile_right = self.get_tile_bounds(i, j)
            cropped_top, cropped_left, cropped_bottom, cropped_right = self.crop_region(i, j, top, left, bottom, right)
            cropped_tile_image = tile.image[..., cropped_top-tile_top:cropped_bottom-tile_top, cropped_left-tile_left:cropped_right-tile_left]
            cropped_tile_weights = self.weights[..., cropped_top-tile_top:cropped_bottom-tile_top, cropped_left-tile_left:cropped_right-tile_left].unsqueeze(0).unsqueeze(0).expand(self.batch_size, -1, -1, -1)
            output[..., cropped_top-top:cropped_bottom-top, cropped_left-left:cropped_right-left] += cropped_tile_image * cropped_tile_weights
            weights[..., cropped_top-top:cropped_bottom-top, cropped_left-left:cropped_right-left] += cropped_tile_weights
                
        return output / weights
        
if __name__ == "__main__":
    from dummy_model import DummyModel
    
    model = DummyModel(sigma_data=0.5)
    
    # Visualize the effect of overlap on the mean image intensity
    for overlap in [16]:
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5, scaling_p=2, scaling_t=0.01)
        sampler = TiledSampler(model, scheduler, overlap=overlap, timesteps=10,
                               boundary=(0, 0, 256, 256), batch_size=4)
        
        mid_region = sampler.get_region(0, 0, 256, 256, upgrade_batch_size=512)
        import matplotlib.pyplot as plt

        for i in range(sampler.batch_size):
            plt.subplot(1, sampler.batch_size, i+1)
            plt.imshow(mid_region[i, 0].cpu().numpy())
            plt.title(f"Batch {i+1}, Std {mid_region[i].std().item():.2f}")
        plt.suptitle(f"Mid Region with Overlap {overlap}")
        plt.show()