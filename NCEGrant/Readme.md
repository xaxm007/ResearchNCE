- visualization.py is the main file that works for new visualization.

### Run this code to process for ennglish captions
```bash
  video_folder=visualization/videos
  output_folder=visualization/output
  pdvc_model_path=save/anet_tsp_pdvc/model-best.pth
  output_language=en
  bash test_and_visualize.sh $video_folder $output_folder $pdvc_model_path $output_language
```

- NotoSansDevanagari.ttf for nepali font
- Nepali language display requires:
  ```bash
      pip install google_trans_new
  ```

### Run this code to process for neapli captions

```bash
  video_folder=visualization/videos
  output_folder=visualization/output
  pdvc_model_path=save/anet_tsp_pdvc/model-best.pth
  output_language=ne
  bash test_and_visualize.sh $video_folder $output_folder $pdvc_model_path $output_language
```
