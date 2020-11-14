# Ulasan Bacaan Paper SimCLR, Google Research, Brain Team 2020

Saya akan mengulas sebuah *paper* yang berjudul [*"Chen et. al., A Simple Framework for Contrastive Learning of Visual Representations, Google Research Brain Team, 2020"*](https://arxiv.org/pdf/2002.05709.pdf).
Berikut ini beberapa poin penting yang saya pelajari dari membaca *paper* ini.

## Ide Inti
Inti dari *paper* ini adalah pembelajaran supervisi mandiri (*self-supervised learning*) untuk representasi visual menggunakan *contrastive learning*.
Dengan menggunakan augmentasi data secara berpasangan yang dimasukkan sebagai input pada *network*, model dilatih untuk memaksimalkan *agreement* di antara pasangan ini dengan menimimalisasi *contrastive loss*. Diharapkan agar model dapat belajar sendiri karakteristik sebuah kategori citra walaupun tidak menggunakan label yang dianotasi oleh manusia.


