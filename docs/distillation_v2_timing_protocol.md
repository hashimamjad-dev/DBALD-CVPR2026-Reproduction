# V2 timing sample, fixed before timing

Measure the first two validation records in manifest order from each dataset, at seed 17, with two repetitions per scene and one initial warmup per method. This is eight timing observations per method across four scenes. Run each method in its own process, serially on the same GPU, with the same cooling guard and synchronization. Methods: teacher20, teacher4, student_v1 and the validation-selected checkpoint of each of the three controlled candidates. This small timing sample estimates prototype latency, not production throughput or a population confidence interval.

Include encoding, initial noise, denoising, decoding, safety checking and final compositing. Exclude model loading, source-file reading, output writing and cooling. Student processes unload the teacher UNet and text encoder, retaining the shared VAE and safety checker. Report peak allocated CUDA memory, not whole-device memory. Record first and second repetitions separately. Do not pool these numbers with v1's differently sampled timing experiment.

This sampling specification does not change the predeclared quality acceptance criteria or validation-only checkpoint selection. Final reused test images remain excluded until the selected checkpoint hash has been written to selection.json.
