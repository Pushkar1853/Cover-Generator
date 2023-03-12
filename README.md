<h1 align="center">
  Music-Cover-generator
</h1>
<h3 align="center">
  🎶 ➡ 🧠  ➡ 🖼️
</h3>

## Description
<img src="https://github.com/Pushkar1853/Music-Cover-generator/blob/5918290f9ce9b4ed905d118e958d6a8ccadf4b8c/images/Abbey-Road-Cover.jpg" alt="your_alt_text" align="right" style="width: 50%; height: 60%;">
* This is a simple application that uses the spectacular Stable Diffusion model to generate images from song lyrics.

* We apply a large multilingual language model in open-ended generation of English song lyrics, and
evaluate the resulting lyrics for coherence and creativity using
human reviewers. 
* We find that current computational metrics for evaluating large language model outputs
have limitations in evaluation of creative writing. We note
that the human concept of creativity requires lyrics to be
both comprehensible and distinctive — and that humans assess certain types of machine-generated lyrics to score more
highly than real lyrics by popular artists.
* Inspired by the inherently multimodal nature of album releases, we leverage
a English-language stable diffusion model to produce high quality lyric-guided album art, demonstrating a creative approach for an artist seeking inspiration for an album or single.

### Generates music album covers using Latest AI tools, namely:
* Stable Diffusion

<img src = "images\stable-diffusion-text-to-image.png" align="center" style="width: 80%; height: auto;">
<img src = "images\stable-diffusion-unet-steps.png" align="center" style="width: 80%; height: auto;">

#### How are lyrics transcribed?
This notebook uses openai's recently released 'whisper' model for performing automatic speech recognition. OpenAI was kind enough to offer several different sizes of this model which each have their own pros and cons. This notebook uses the largest whisper model for transcribing the actual lyrics. Additionally, we use the smallest model for performing the lyric segmentation. Neither of these models is perfect, but the results so far seem pretty decent.

* OpenAI Whisper for transcript
  <img src = "images\whisper.png" style="width: 80%; height: auto;">
* Spotify Access token for songs retrieval

  <img src ="images\spotify1.png"  style: height="180px" width="700px" align="center" >
* Genius Lyrics for songs

<img src ="images\genius1.png"  style: height="200px" width="670px" align="center" >

### Papers reviewed:
* [In BLOOM: Creativity and Affinity in Artificial Lyrics and Art](https://www.researchgate.net/publication/367165610_In_BLOOM_Creativity_and_Affinity_in_Artificial_Lyrics_and_Art)
<img src="https://github.com/Pushkar1853/Music-Cover-generator/blob/eb1c8fc1bd521b27116554f39df0891aa988189d/images/chin1.png" style="width: 50%; height: 60%;">

* [GLIGEN, Open-Set Grounded Text-to-Image Generation](https://www.researchgate.net/publication/367216711_GLIGEN_Open-Set_Grounded_Text-to-Image_Generation) 

Large-scale text-to-image diffusion models have madeamazing advances. However, the status quo is to usetext input alone, which can impede controllability. In thiswork, we propose GLIGEN,Grounded-Language-to-ImageGeneration, a novel approach that builds upon and extendsthe functionality of existing pre-trained text-to-image dif-fusion models by enabling them to also be conditioned ongrounding inputs. To preserve the vast concept knowledge ofthe pre-trained model, we freeze all of its weights and injectthe grounding information into new trainable layers via agated mechanism. Our model achieves open-world groundedtext2img generation with caption and bounding box condi-tion inputs, and the grounding ability generalizes well tonovel spatial conﬁgurations and concepts. GLIGEN’s zero-shot performance on COCO and LVIS outperforms existingsupervised layout-to-image baselines by a large margin

<img src="https://github.com/Pushkar1853/Music-Cover-generator/blob/eb1c8fc1bd521b27116554f39df0891aa988189d/images/pap2.png">






