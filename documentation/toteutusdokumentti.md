## Ohjelman rakenne

Ohjelma koostuu erinäisistä .py-tiedostoista kansiossa src/eigenface

### Päämoduulit:
- `cli.py`: command line interface -komennot eigenface-mallin kouluttamiselle ja kuvan testaamiselle
- `dataset.py`: kuvatiedostojen haku ja muunto matriisiksi
- `matrix.py`: matriisioperaatiot ilman valmiita kirjastoja
- `vector.py`: vektorilaskennan operaatiot toteutettuna ilman valmiita kirjastoja
- `pca.py`: PCA eli pääkomponenttianalyysin funktiot ja algoritmin koulutus
- `qr.py`: QR-hajotelman funktiot ominaisarvojen laskentaan

### Muut moduulit:
- `app.py`: tarkoitettu lähinnä projektin debuggaamiseen eikä ole osa lopullista käyttöjärjestelmää

---

## Toiminta

Eigenface-algoritmi perustuu pääkomponenttianalyysiin

Vaiheittain:

1. **Datan lataus**  
   dataset.py lukee kuvat annetusta kansiosta, muuntaa ne harmaasävyisiksi, muuntaa jokaisen vektoriksi ja yhdistää ne matriisiksi

2. **Datan käsittely**  
   Luodulle matriisille lasketaan keskiarvo ja poistetaan se jokaisesta sarakkeesta eli keskitetään data

3. **Kovarianssimatriisi**  
   Keskitetylle datalle lasketaan kovarianssimatriisi

4. **Ominaisarvot ja -vektorit**  
   Kovarianssimatriisi puretaan sen ominaisarvoihin ja vektoreihin

5. **Eigenface-vektorit**  
   Valitaan top k suurimman ominaisarvon omaavat ominaisvektorit

6. **Projektio**  
   Jokainen keskitetty kuva projisoidaan valittuihin ominaisvektoreihin. Näistä luodaan matriisi, jossa yksi sarake on yhden kuvan painot

7. **Ennustus**  
   Halutusta testikuvasta muodostetaan samanlainen harmaasävyinen vektori, ja se projisoidaan samaan eigenface-avaruuteen kuin koulutuskuvat.  
   Tämän jälkeen verrataan laskettua painovektoria muihin painovektoreihin laskemalla niiden välinen etäisyys.  
   Jos etäisyys on liian suuri, kuvaa ei lasketa tunnistetuksi.

---

## Aika- ja tilavaativuudet

Olkoon n kuvien määrä ja kuvien koko k × k. Tällöin yksi kuva vektorina on kooltaan k².

- Kuvien muuttaminen vektoreiksi:  
  Jokainen kuva muunnetaan kerran vektoriksi = O(n · k²), koska jokainen pikseli käsitellään kerran per kuva.

- Keskiarvovektorin laskenta (compute_mean):  
  Käydään kaikki pikselit läpi = O(n · k²)  
  (Summataan jokaisen kuvan kaikki komponentit.)

- Datan keskitys (center_data):  
  Jokaisesta kuvasta vähennetään keskiarvo = O(n · k²). Jokaiselle pikselille tehdään yksi operaatio.

- Kovarianssimatriisi:  
  Lasketaan matriisikertolaskuna = O(n² · k²)  
  (Kertolaskussa käydään läpi kaikki kuvaparit (n²), ja jokaisessa summataan k² termiä.)

- QR-hajotelma:  
  n × n -matriisille yhden hajotelman aikavaativuus on O(n³), joten  
  iteratiivisesti ominaisarvojen ratkaisu = O(I · n³), missä I on iteraatioiden määrä  

- Projektio eigenface-avaruuteen:  
  m eigenfacea, n kuvaa, vektorin koko k² = O(n · m · k²), jossa jokainen kuva projisoidaan m komponenttiin pistetulojen avulla.

Kokonaisaikavaativuus eigenface-mallin koulutukselle olisi:

3 · O(n · k²) + O(n² · k²) + O(I · n³) + O(n · m · k²)

Joka voidaan yksinkertaistaa:

O(n² · k² + I · n³)

---

# Tilavaativuudet

Olkoon n kuvien määrä ja kuvien koko k × k. Tällöin yksi kuva vektorina on kooltaan k².

- Kuvat:  
  Meillä on n määrä kuvia, ja jokainen on kokoa k² = O(n · k²)

- Keskiarvovektori:  
  Yksi vektori kokoa k² = O(k²)

- Keskitetty data:  
  Sama kuin kuvat eli O(n · k²)

- Kovarianssimatriisi:  
  Koska tässä toteutuksessa käytetään pienempää kovarianssimatriisia, koko on O(n²)

- Top k eigenfacea:  
  Olkoon m eigenfacejen määrä, niin tilavaativuus = O(m · k²)

- Projektiot:  
  n vektoria, joista jokainen on kokoa m = O(n · m)

Kokonaistilavaativuus eigenface-toteutukselle:

O(n · k²) + O(n²) + O(m · k²) + O(n · m)

Joka voidaan yksinkertaistaa:

O(n · k² + n²)

---

## Mahdolliset puutteet ja parannusehdotukset

Laskennallisesti raskain osa toteutuksessa on ominaisarvojen laskenta QR-hajotelman avulla. Tämä muodostaa suuren osan algoritmin aikavaativuudesta.  
QR-hajotelmalle olisi mahdollisesti voinut olla tehokkaampi korvike.

---

## Kielimallien käyttö

Työssä on käytetty ChatGPT-kielimallia seuraaviin tarkoituksiin:

- dokumentaation kieliasun korjaaminen  
- käsitteiden selventäminen  
- tarkentavien kysymysten esittäminen erityisesti QR-hajotelmaan liittyen, joka ei ollut ennestään tuttu, sekä pienten teknisten ongelmien ratkaisuun

Kielimallia ei käytetty varsinaisen algoritmikoodin toteuttamiseen.

---

## Lähteet

- Eri kirjastojen dokumentaatio (os, argparse, pathlib, Pillow)
- Wikipedia-artikkelit QR-hajotelmasta ja PCA:sta
- [Raportti Al Akhawayn yliopistosta](https://cdn.aui.ma/sse-capstone-repository/pdf/spring-2023/Face%20Recognitionand%20Detection%20Using%20Eigenface%20Algorithm%20Final%20Report.pdf)  
  Hyödyllinen erityisesti alussa ymmärtämään, mitä ollaan tekemässä
