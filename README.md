# LunarTTS-2
LunarTTS 2 is a WIP FastSpeech2 modification that adds Text Predicting GSTs influenced by BERT to create a Context-aware, emotion-conditioned FastSpeech2 Architecture

This model uses Hierarchical GST with contextual embeddings from RoBERTa to infer style from text directly

A TPSE module (lightweight FC + GRU) learns to map RoBERTa embeddings to GST latent space, enabling implicit emotion from text only.



> Keep in mind that this repo is still really in early development (I started working on it less than a week ago as of 5.5.2026)
> Expect unoptimized code, possibly broken code, and constant refactoring

# Contribuiting
All PRs are welcome, but make sure you:
- Test your feature so that it actually works
- Do not vibecode ffs
- Try to make your code clean
- Make sure you describe whatever PR you are working on

  
If you have a question or an issue, then make an issue on it and I _should_ respond

# Pretrain model
TBA
There will be a release once I feel the architecture is stable and mature enough.
If you want, go ahead and train your own, but this repo is constantly changing, and your model could be out of date.

# Updates
5.5.2026 - Uploaded to GitHub
