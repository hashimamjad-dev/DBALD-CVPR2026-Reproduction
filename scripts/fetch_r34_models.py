import fetch_overnight_models as fetcher

fetcher.ASSETS={
 'adnet_tusimple_res34.pth':'https://github.com/Sephirex-X/ADNet/releases/download/ckpt/tusimple_res34.pth',
 'adnet_culane_res34.pth':'https://github.com/Sephirex-X/ADNet/releases/download/ckpt/culane_res34.pth',
 'resa_r34_tusimple.zip':'https://github.com/Turoad/lanedet/releases/download/1.0/resa_r34_tusimple.zip'}
# Keep this download's manifest separate from CULane assets.
if __name__=='__main__':
    fetcher.main('results/r34_checkpoints.json')
