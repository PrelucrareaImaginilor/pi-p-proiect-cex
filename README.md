# Detecția și recunoașterea limbajului semnelor din imagini/ secvențe video
CEX, Epure Carla-Maria și Velnic Vlad-Andrei

# 1. Hand Gesture Recognition using Image Processing and Feature Extraction Techniques - Ashish Sharma, Anmol Mittal, Savitoj Singh, Vasudev Awatramani

Aplicație/ Domeniu: Recunoașterea gesturilor mâinii pentru interpretarea Limbajului American al Semnelor (ASL) prin imagini statice. Se urmărește accesibilizarea limbajului semnelor, utilizatorii nefiind nevoiți să aibă cunoștințe în prealabil.

Tehnologii utilizate: Nu sunt menționate echipamente hardware. Se pune accentul pe faptul că majoritatea seturilor de date disponibile sunt lipsite de zgomot, iar imaginile făcute în viața reală nu corespund în totalitate acestui aspect, fiind nevoie de un proces lung de pre-procesare a imaginii. Ca și tehnologii software se folosesc: Canny Edge Detection- extragerea marginilor, ORB (Oriented FAST and Rotated BRIEF)- identificarea mânii și a caracteristicilor și Bag of Words- crearea unui dicționar și a histogramei necesare pentru descrierea trăsăturilor importante.

Metodologie: Se folosesc 4 pași pentru procesarea imaginii- segmentare, extragerea trăsăturilor, generarea histogramei dicționarului și clasificarea. În procesul de segmentare imaginea se transformă în grayscale și se folosește Canny Edge Detection pentru reducerea zgomotului de fundal. Se utilizează ORB pentru detectarea caracteristicilor, urmate de generarea vocabularului și a histogramei aferente. 

Rezultate: Acuratețea cea mai mare a fost obținută de modelul ORB în combinație cu Multilayer Perceptron (MLP)- 96.96% și cu K-Nearest Neighbours (KNN)- 95.96%.

Limitări: Implementarea recunoaște doar gesturi statice. Sistemul nu a fost testat pe imagini RGBD care conțin date despre adâncime.

Comentarii: Bază bună pentru proiect, descrie limitările care trebuie depășite și metodologii cu rezultate bune pentru imagini statice. Poate fi extins pentru componenta de analiză video, alături de utilizarea rețelelor neuronale.


# 2. Alphabet Recognition of Sign Language Using Machine Learning - AVINASH KUMAR SHARMA, ABHYUDAYA MITTAL, AASHNA KAPOOR, ADITI TIWARI

Domeniu: Recunoașterea limbajului semnelor atât pentru ASL, cât și pentru ISL- Indian Sign Language. Se dorește captarea gesturilor și transformarea acestora în text și sunet.

Tehnologii utilizate: Se folosește camera web alături de biblioteca OpenCV. Proiectul se bazează pe o rețea neuronală și pe reantrenarea unui model deja antrenat (Inception V3) pe un set nou de date. Pentru convertirea în fișiere audio se folosește Google Text-to-Speech (gTTS).

Metodologie: Proiectul a fost împărțit în 6 etape- colectarea bazei de date, antrenarea modelului, pre-procesare și segmentarea mâinii, extracția trăsăturilor, clasificarea și convertirea text to speech. În loc să se construiască un model de la zero, s-a reantrenat modelul Inception V3 folosind seturi de date publice de pe Kaggle. Etapele de pre-procesare și de extragere a trăsăturilor sunt realizate într-o manieră automată cu ajutorul modelului Inception V3. Cu ajutorul gTTS se convertește textul procesat și se generează un fișier audio.

Rezultate: Folosirea modelului Inception V3 a oferit rezultate foarte bune, având o acuratețe de 98.99%. Lucrarea a rezolvat o problemă comună din trecut: literele C, L, M, N, R, U, Y nu erau recunoscute.

Limitări: Incapacitatea sistemului de a recunoaște cuvinte întregi, ci doar litere individuale.

Comentarii: Articolul are o relevanță ridicată pentru proiectul nostru deoarece se folosește un model deja antrenat care atinge o rată de acuratețe foarte bună. Se poate extinde cu analiza secvențelor video.


# 3. A Comprehensive Review of Sign Language Recognition: Different Types, Modalities, and Datasets- M. MADHIARASAN, PARTHA PRATIM ROY

Domeniu: Lucrarea reprezinta o analiză a realizărilor în domeniul recunoașterii limbajului semnelor, tehnologiile utilizate, separarea lor pe fiecare dialect și segregarea lor în recunoaștere izolată sau continuă.

Tehnologii utilizate: Sunt prezentate tehnologiile de rețele neuronale folosite precum CNN(Convolutional Neural Network), LSTM (Long Short-Term Memory) sau hibride, dar și aspectul achiziției de date, posibilă atât prin camere video, cât și prin senzori precum mănuși kinetice.

Metodologie: Procesul general include colectarea de date (video/senzori), preprocesare, extragerea caracteristicilor, clasificare și evaluare, combinând abordări vizuale și bazate pe senzori pentru a acoperi semne manuale și faciale.

Rezultate: Modelele de vârf ajung la rate de acuratețe foarte ridicate în recunoașterea semnelor izolate (de ex. 99-98%), însă performanța scade pentru recunoașterea continuă sau pentru date nestructurate(de ex. 90-85%).

Limitări: Printre limitări se numără lipsa unor seturi de date mari și variate, dificultatea generalizării între semnatari, problemele cauzate de variații de iluminare, fundal și occluziuni, precum și diferențele semnificative față de dialecte

Comentarii: Lucrarea subliniază că domeniul este în continuă dezvoltare, fiind necesare dataseturi mai bune și modele multimodale mai robuste. Din această lucrare tragem concluzia că vom fi limitați de metoda de achiziție de date, deoarece vom folosi doar camera video.


# 4. Application of transfer learning to sign language recognition using an inflated 3D deep convolutional neural network- ROMAN TOENGI

Domeniu: Lucrarea tratează recunoașterea limbajului semnelor cu accent pe aplicarea transferului de învățare la recunoașterea semnelor izolate din diferite limbi, în contextul inteligenței artificiale și viziunii computerizate.

Tehnologii utilizate: Studiul utilizează rețele neuronale convoluționale tridimensionale bazate pe TensorFlow si Inception-v3, pentru a transfera cunoștințe între modele specifice ASL și SIGNUM (German Sign Language).

Metodologie: Se implementează un model care este pre-antrenat pe setul ASL și apoi perfecționat pe diferite subseturi de date SIGNUM, evaluând transferabilitatea cunoștințelor între două limbi ale semnelor cu ajutorul experimentelor controlate pe date de dimensiuni diferite.

Rezultate: Modelele ce folosesc transfer learning au obținut o creștere semnificativă a acurateții pe setul țintă comparativ cu modelele fără pre-antrenare, obținând îmbunătățiri de până la 21% în funcție de mărimea setului de antrenament, cu avantaje și în ceea ce privește viteza de convergență și generalizarea.

Limitari: Este subliniată dependența rezultatului final de natura datelor de pre-antrenare, de asemenea, lucrarea se concentrează doar pe detectarea semnelor izolate și nu a secvențelor continue.

Comentarii: Rezultatele demonstrează avantajele transfer learning-ului, dar si nevoia de fine tuning pe data set-uri specifice fiecarui dialect. Lucrarea este relevantă pentru adaptarea unui model pre-antrenat pentru ASL la unul specific LSR.


# 5. Score-level Multi Cue Fusion for Sign Language Recognition- Cagrı Gokce,  Ogulcan Ozdemir, Ahmet Alp Kındıroglu, Lale Akarun

Domeniu: Lucrarea se concentrează pe recunoașterea limbajului semnelor izolate prin analiza vizuală și fuzionarea multiplilor indicatori, în special pentru limba semnelor turcești, în domeniul viziunii computerizate și recunoașterii acțiunilor vizuale Sign Language Recognition﻿.

Tehnologii utilizate: Se utilizează rețele neuronale convoluționale reziduale tridimensionale (3D Residual CNN) cu convoluții mixte, selecție spațială și temporală a regiunilor relevante (mâini, față, corp superior), și fuzionarea rezultatelor la nivel de scor pentru îmbunătățirea performanței.

Metodologie: Abordarea implică antrenarea separată a modelelor specializate pe diferite regiuni (manuale și non-manuale), selectarea cadrului temporal activ al semnelor (momentul mișcării mâinilor), apoi combinarea rezultatelor fiecărui model prin fuziune ponderată a scorurilor pentru o decizie finală.

Rezultate: Metoda a dus la o acuratețe de 94.94% pe un set de date extins cu 22.000 de clipuri video, superând metodele anterioare și evidențiind că fuzionarea indicilor din diferite regiuni vizuale crește remarcabil performanța recunoașterii.

Limitări: Performanța scade în cazul semnelor cu gesturi repetitive sau monomorfemice și în cele cu mișcări circulare complexe, iar modelul este testat doar pe semne izolate, nu pe propoziții semnalizate continuu.

Comentarii: Studiul sugerează că pentru a îmbunătăți recunoașterea, este nevoie de creșterea rezoluției pentru recunoașterea formei mâinii și dezvoltarea unor rețele mai profunde sau a unor metode de optimizare superioare; astfel, această cercetare promovează dezvoltarea semnalizării automate și a traducerii limbajului semnelor.​



## Proiectarea soluţiei:

Ca și abordare inițială vom analiza doar imaginile statice. Din articolele studiate, am extras cei mai importanți pași pentru realizarea unui program de recunoaștere a limbajului semnelor. 

![WhatsApp Image 2025-10-30 at 11 50 39](https://github.com/user-attachments/assets/8f4e1105-8ca1-4d82-aaa5-d7dbd7d001f1)

1. Achiziția datelor: În acest stadiu se colectează imaginile care conțin datele pentru recunoașterea limbajului semnelor. Se folosește camera web și biblioteca OpenCV pentru citirea fluxului.
2. Preprocesarea imaginilor: Acesta este un stadiu important pentru un model robust- se pregătesc imaginile brute pentru a putea fi utilizate de modelul AI, uniformizând datele. Se pot folosi OpenCV, MediaPipe.
Etape cheie:

  &nbsp;&nbsp;&nbsp;&nbsp;2.1. Redimensionarea imaginii  
  &nbsp;&nbsp;&nbsp;&nbsp;2.2. Conversie in grayscale  
  &nbsp;&nbsp;&nbsp;&nbsp;2.3. Eliminarea zgomotului prin filtrre (Gaussian blur, threshold)
  &nbsp;&nbsp;&nbsp;&nbsp;2.4. Detectia mainii, conturului

3. Extragerea caracteristicilor: O soluție foarte bună de tracking a mâinii și de extragere a punctelor importante este MediaPipe Hands. Se obține astfel un vector ce reprezintă caracteristicile geometrice ale mâinii- forma și poziția. 
4. Modelul AI: Primește vectorul rezultat în pasul anterior și imaginile după preprocesare și învață să recunoască semnele.
5. Postprocesarea și recunoașterea: Convertește rezultatele modelului în eticheta corespunzătoare literei. 
6. Afișare rezultat: Posibilitatea de a salva output-ul într-un fișier separat, conversie în vorbire.

