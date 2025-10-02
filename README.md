# Monet-Style Image Generation (CycleGAN)

A notebook project that trains a **CycleGAN** to translate between **landscape photos** and **Claude Monet paintings**. The model learns unpaired image-to-image translation using generators and discriminators with cycle consistency.

## Results

* Trained for 25 epochs on Kaggle’s Monet dataset (~300 paintings) and ~7k landscape photos.
* End of training losses (epoch 25):

  * Generator G (Photo→Monet): 3.39
  * Generator F (Monet→Photo): 3.37
  * Discriminator Monet (D_Y): 0.58
  * Discriminator Photo (D_X): 0.59
  * Cycle loss: 1.67
  * Identity loss: 0.77
* Generated samples showed realistic Monet-like brush strokes and colors.

## Approach

1. **Data:** Monet paintings (~300) + landscape photos (~7k), preprocessed to 256×256, normalized to [-1,1].
2. **Architecture:**

   * Generators: U-Net style encoder–decoder with skip connections.
   * Discriminators: PatchGAN (70×70 patches) for fine detail realism.
3. **Losses:**

   * Adversarial (fooling discriminators).
   * Cycle consistency (Photo→Monet→Photo ≈ original).
   * Identity (Monet→Monet ≈ Monet, weight 0.5).
4. **Training:**

   * Adam optimizer (lr=2e-4, β1=0.5).
   * tf.data pipelines for efficient loading + augmentation (jitter, flips).
   * Checkpoints saved every 5 epochs.
