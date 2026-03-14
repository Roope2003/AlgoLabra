## Ohjelmointikieli

Työssä käytetään ohjelmointikielenä Pythonia.

Työssä toteutetaan kasvojentunnistus Eigenface-menetelmällä.
Kaikki matriisioperaatiot toteutetaan itse ilman valmiita kirjastoja, joten työssä toteutetaan useita matriisilaskentaan liittyviä algoritmeja.
Näistä pitää toteuttaa ainakin:
* Ylieiset matriisioperaatiot (kertolaskut, transpoosi)
* Ominaisarvot ja ominaisvektorit
* Kovarianssimatriisi


Työn tavoitteena on ratkaista kasvojentunnistusongelma. Ohjelma oppii joukon kasvokuvia ja tunnistaa uuden kuvan vertaamalla sitä opittuihin kuviin.

Tietorakenteina tulee toimimaan Vektorit ja Matriisit eli Listat ja 2D-listat



## Syötteet

Ohjelma saa syötteenä harmaasävykuvia ihmisten kasvoista.  
Kaikki kuvat ovat saman kokoisia.

Jokainen 2D-kuva muutetaan 1D-vektoriksi.  
Näistä vektoreista muodostetaan matriisi, jota käytetään laskennassa.

Syötteenä annetaan

- joukko opetuskuvia
- testikuva




## Aikavaativuus

Olkoon M kuvien määrä ja N yhden kuvan pikselien määrä.  
Kuvista muodostetaan matriisi A, jonka koko on N × M.

Kuvien muuttaminen vektorieksi vie ajan O(M*N).  
Kovarianssimatriisin laskeminen tehdään kertolaskuna kaavalla  AᵀA, joten aikavaativuus on O(M²N).  
Ominaisarvojen laskeminen kovarianssimatriisista vie ajan O(M³).  







## Harjoitustyön ydin

Työn ydin on toteuttaa Eigenface-menetelmän matemaattinen osa ilman valmiita kirjastoja.  
Keskeinen osa on kovarianssimatriisin muodostaminen, ominaisarvojen laskeminen ja pääkomponenttianalyysin käyttö kasvojentunnistuksessa.

Suurin osa työajasta käytetään lineaarialgebran algoritmien toteuttamiseen


Kurssin dokumentaatio kirjoitetaan suomeksi.
Suoritan tietojenkäsittelytieteen kandin tutkintoa.


## Mahdolliset hyödylliset lähteet



- https://en.wikipedia.org/wiki/Eigenface
- https://en.wikipedia.org/wiki/Principal_component_analysis
- https://en.wikipedia.org/wiki/Covariance_matrix
- https://www.face-rec.org/algorithms/PCA/jcn.pdf "Turk, M. & Pentland, A. (1991). Eigenfaces for Recognition"

