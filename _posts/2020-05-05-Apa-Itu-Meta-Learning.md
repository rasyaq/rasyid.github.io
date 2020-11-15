# Apa Itu *Meta-Learning*

*Meta-learning* adalah sebuah mekanisme pembelajaran *machine learning* untuk mempelajari sebuah *task* baru yang tidak pernah dipelajari sebelumnya. Istilah lainnya, *learning to learn*. 

Yang dimaksud dengan *Learning to learn* di sini adalah, pada tahap *training* (pelatihan), model dipaparkan pada beberapa *training tasks* dengan set data yang berbeda berisikan *support* dan *query* yang berisi data sebanyak N-kelas dan K-sample (dikenal juga dengan *N-way K-shot*). Lalu, *training* dilakukan dengan bertahap di mana pada setiap taha, model parameter diperbarui pada *training task* yang dipilih secara acak. 

Tujuan utama *Meta-Learning* adalah agar model bisa belajar pemisahan kelas secara umum dibanding hanya berfokus pada pemisahan subset kelas objek yang lebih spesifik. 

*Few-shot learning*, atau pembelajaran model dengan data yang terbatas khususnya pada kasus distribusi *long-tailed*, biasa dikategorikan juga sebagai salah satu bentuk *Meta-Learning*.

Model *meta-learning* diharapkan mampu beradaptasi atau menggeneralisasi *task* baru dan lingkungan baru yang belum pernah ditemukan selama waktu *training*. Proses adaptasi model diharapkan agar model dapat menyelesaikan *task* baru dengan baik.

Model meta-learning yang baik harus dilatih di atas berbagai *task* dan dioptimalkan untuk kinerja terbaik pada distribusi *task* yang ada, termasuk *task* yang belum teramati. Setiap *task* dikaitkan dengan dataset D, yang berisikan vektor fitur dan labelnya. Parameter model optimal adalah:
![\theta^*=\argmin_\theta E_{D~p(D)}\[L_{\theta}(D)\]](https://render.githubusercontent.com/render/math?math=%5Ctheta%3D%5Cargmin_%5Ctheta%20E_%7BD~p(D)%7D%5BL_%7B%5Ctheta%7D(D)%5D)
