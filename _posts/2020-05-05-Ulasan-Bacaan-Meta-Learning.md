# Meta Learning

Meta-learning adalah sebuah mekanisme pembelajaran *machine learning* untuk mempelajari sebuah *task* baru yang tidak pernah dipelajari sebelumnya. Istilah lainnya, *learning to learn*.

Model meta-learning yang baik diharapkan mampu beradaptasi dengan baik atau menggeneralisasi *task*  baru dan lingkungan baru yang belum pernah ditemukan selama waktu *training* (pelatihan). Proses adaptasi model, yang pada dasarnya adalah sesi *training* yang lebih kecil, terjadi selama *testing* tetapi dengan paparan terbatas terhadap konfigurasi *task* baru. Akhirnya, model yang diadaptasi dapat menyelesaikan *task* baru dengan baik.

Model meta-learning yang baik harus dilatih di atas berbagai *task* pembelajaran dan dioptimalkan untuk kinerja terbaik pada distribusi *task*, termasuk *task* yang belum teramati. Setiap *task* dikaitkan dengan dataset D, yang berisi vektor fitur dan label sebenarnya. Parameter model optimal adalah:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta=\argmin{\theta}{E_{D~p(D)}[L_{\theta}(D)]"/>
