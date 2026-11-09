"""CPU-safe tests for ultraedit_trigger.py's pure geometry (window fitting, mask construction,
compositing) -- the parts that don't need a GPU or the actual UltraEdit weights. The real model
load/inference path (UltraEditTrigger.__init__, feasibility_check) is exercised only by a human
running feasibility_check() on a CUDA machine; see that function's own docstring.
"""
import numpy as np
import pytest
from lane_repro.ultraedit_trigger import UltraEditTrigger

class FakePipe:
    """Stands in for the diffusers pipeline: records its call args and returns the input mask
    region recoloured, so a test can check exactly which pixels were sent for editing."""
    def __call__(self,prompt,image,mask_img,negative_prompt,num_inference_steps,
            image_guidance_scale,guidance_scale,generator):
        self.last_call=dict(prompt=prompt,mask=np.asarray(mask_img),steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,guidance_scale=guidance_scale)
        from PIL import Image
        array=np.asarray(image).copy()
        array[np.asarray(mask_img)>0]=255
        class Result: images=[Image.fromarray(array)]
        return Result()
    def to(self,device): return self
    def enable_attention_slicing(self): pass
    def enable_model_cpu_offload(self): pass

def make_trigger(edit_window=64):
    trigger=object.__new__(UltraEditTrigger)
    trigger.pipe=FakePipe(); trigger.edit_window=edit_window; trigger.steps=10
    trigger.image_guidance_scale=1.5; trigger.guidance_scale=7.5; trigger.drop_t5=True
    return trigger

def test_call_masks_exactly_the_trigger_footprint_within_the_window():
    trigger=make_trigger(edit_window=64)
    image=np.zeros((200,200,3),np.uint8)
    result=trigger(image,100,100,10,'Add a small and sparse brown mud',seed=0)
    assert result.shape==image.shape
    assert result[100:110,100:110].min()==255  # trigger footprint recoloured
    assert result[0,0].sum()==0 and result[199,199].sum()==0  # far outside the window untouched

def test_call_window_clamps_to_image_bounds_near_an_edge():
    trigger=make_trigger(edit_window=64)
    image=np.zeros((200,200,3),np.uint8)
    result=trigger(image,5,5,10,'x',seed=0)  # near top-left corner
    assert result[5:15,5:15].min()==255

def test_call_rejects_trigger_larger_than_edit_window():
    trigger=make_trigger(edit_window=32)
    image=np.zeros((200,200,3),np.uint8)
    with pytest.raises(ValueError): trigger(image,100,100,40,'x',seed=0)

def test_call_passes_prompt_and_settings_through():
    trigger=make_trigger(edit_window=64)
    image=np.zeros((200,200,3),np.uint8)
    trigger(image,100,100,10,'Add a small traffic cone',seed=0)
    assert trigger.pipe.last_call['prompt']=='Add a small traffic cone'
    assert trigger.pipe.last_call['steps']==10
    assert trigger.pipe.last_call['mask'].sum()>0
    mask = trigger.pipe.last_call['mask']
    assert mask.shape == (64, 64, 3)
    assert np.array_equal(mask[:, :, 0], mask[:, :, 1])
    assert np.array_equal(mask[:, :, 0], mask[:, :, 2])
    assert np.count_nonzero(mask[:, :, 0]) == 10 * 10

def test_compositing_preserves_outside_pixels_even_when_pipeline_edits_entire_crop():
    from PIL import Image
    from types import SimpleNamespace

    class DriftingPipe:
        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(images=[Image.fromarray(np.full((64, 64, 3), 255, np.uint8))])

    trigger = make_trigger(edit_window=64)
    trigger.pipe = DriftingPipe()
    image = np.full((100, 100, 3), 17, np.uint8)
    result = trigger(image, 40, 40, 10, 'fixture', seed=0)
    outside = np.ones((100, 100), dtype=bool)
    outside[40:50, 40:50] = False
    assert np.array_equal(result[outside], image[outside])
    assert np.all(result[40:50, 40:50] == 255)
