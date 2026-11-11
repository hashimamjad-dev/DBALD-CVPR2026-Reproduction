# Verified UltraEdit standalone run

Kaggle notebook hashimamjad03/dbald-research version 2 completed the original BleachNick/SD3_UltraEdit_w_mask pipeline. Results were recovered on 2026-11-09. The exact submitted source and metadata are preserved under submitted/.

Model revision: e5beecb69a3ef1399718be2f99b61730b8bf0346
UltraEdit fork revision: 1af5f0478d56d62cf6f35781a2ab63e03d656b2d

Internet was enabled. Public checkpoint access succeeded without a Hugging Face token. The run found no existing compatible cache or attached snapshot. It downloaded 31,005,015,718 bytes to one Hugging Face cache under /kaggle/temp and verified 30 files against repository SHA-256 or Git blob hashes. checkpoint_files.json retains the verification manifest. The weights were temporary session assets and their continued availability in a future session is unverified.

All three text encoders were retained. The pipeline used FP16 and whole-model CPU offloading to GPU 0. The upstream horse input was resized to 512 by 512, with a white full-image mask and prompt "What if the horse wears a hat?". Seed 0, 50 inference steps, image guidance 1.5 and guidance 7.5 were used.

Measured loading time was 78.412445 seconds. Inference took 58.719186 seconds. Total recorded execution took 496.880052 seconds. Peak process RSS sampled during loading and inference was 23,386,152,960 bytes (21.7801 GiB). Peak PyTorch CUDA allocation was 12,949,083,648 bytes (12.0598 GiB), with 13,077,839,872 bytes reserved. These GPU counters exclude driver and other-process allocations. The RAM monitor sampled every 0.5 seconds.

edited.png is the generated output. Its SHA-256 is 5434e27624ef841b4d5f4e79ff8b53e210e8ba586847f2a076179cd5956f455c. Recovery verified its hash, dimensions and difference from the input. Visual inspection confirms a hat on the horse. The original input, mask, generated image, package versions and execution log are preserved.

The reusable source in scripts/kaggle/ultraedit_standalone.py now explicitly pins the successful model revision. scripts/kaggle/ultraedit.ipynb and configs/kaggle_ultraedit.json provide the corresponding notebook and submission metadata. The original version 2 resolved main and then downloaded its immutable revision. That executed source is preserved unchanged in submitted/.

Table 1 was refreshed using the existing manifest and measurements. It accepted 15 evidence records, rejected none and exported all 220 displayed cells. No full detector inference was repeated. LaneATT R34's existing full TuSimple clean ACC remains 95.63477799459051 percent, a difference of -1.1752220054094948 points from the 96.81 percent published benign reference. This image edit supplies component execution evidence only and contributes no ACC or ASR measurement. All method-specific poisoned-model results remain unmeasured.

Validation: 11 Table 1 tests passed. Exact downloaded image SHA-256, 512 by 512 dimensions, checkpoint manifest file count and byte total, and saved internet setting passed local checks. Full checkpoint bytes were verified on Kaggle; weights were not copied locally.

To run again, put ultraedit.ipynb beside a copy of configs/kaggle_ultraedit.json named kernel-metadata.json and submit to the existing notebook. The notebook checks attached resources and caches before downloading. A fresh session may need to download the weights again.
