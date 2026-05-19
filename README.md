# LunarTTS-2
## What is the branch doing?
It separates training into 2 Stages:

- **Stage 1**:
  - Train HGST + FastSpeech2 until HGST is good and FastSpeech2 has converged

- **Stage 2**:
  - Teach TPSE to map BERT outputs to HGST outputs.

>The reason is to simplify training, ensure TPSE learns properly, and to have quicker iteration for experimentation.
>This idea is taken from [TEMOTTS](https://arxiv.org/abs/2405.11413). They have similar ideas to Lunar-TTS (except the use of RoBERTa + TPSE, Princess Luna Branding, HGSTs)


---

LunarTTS 2 is a WIP FastSpeech2 modification that adds Text Predicting GSTs influenced by RoBERTa to create a Context-aware, emotion-conditioned FastSpeech2 Architecture

This model uses Hierarchical GST (HGST) with contextual embeddings from RoBERTa to infer style from text directly

A TPSE module (lightweight FC + GRU) learns to map RoBERTa embeddings to the GST latent space, enabling implicit emotion inference from text alone.



> Keep in mind that this repo is still really in early development (I started working on it less than a week ago as of 5.5.2026)
> Expect unoptimized code, possibly broken code, and constant refactoring

---

# Contribuiting
All PRs are welcome, but make sure you:
- Test your feature so that it actually works
- Do not vibecode, please.
- Try to make your code clean
- Make sure you describe whatever PR you are working on


If you have a question or an issue, then make an issue on it and I _should_ respond

---


# Pretrain model
**TBA**


There will be a release once I feel the architecture is stable and mature enough.
Feel free to train your own, but this repo is constantly changing, and your model could be out of date.

---

# Updates
**5.5.2026** 
- Uploaded to GitHub

**8.5.2026**
- GST + TPSE rework for better expressivity
- TPSE loss is now a hybrid L1 cosine loss system `l1_loss * 0.3 + cosine_loss * 0.7`
- New roBERTa conditioning pipeline
- General Cleanup

---

# Credits
https://github.com/lightbooster/TP-GST-BERT-Tacotron2 - TPSE implementation, some ideas, and GST implementation

https://github.com/ming024/FastSpeech2 - FastSpeech2 implementation I used

https://rf5.github.io/2022/10/18/hgst.html - Guidance on HGSTs


