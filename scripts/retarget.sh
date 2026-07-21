python retargeting/retarget.py \
  --model "/home/tonyroumi/Desktop/move-it-move-it/retargeting/outputs/character2_to_Rub/2026-03-21/22-32-10/skeletal_gan_epoch10500.pt" \
  --source-skeleton retargeting/data/skeletons/character2.npz \
  --target-skeleton retargeting/data/skeletons/Rub.npz \
  --motion /home/tonyroumi/Desktop/move-it-move-it/retargeting/data/bandai/processed/character2/dataset-2_raise-up-left-hand_feminine_051.npz\
  --output-dir outputs/retargeting/character2toRub