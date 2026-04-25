## Kattavuusraportti

```
Name                        Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------
src/eigenface/__init__.py       0      0      0      0   100%
src/eigenface/matrix.py        61      0     38      0   100%
src/eigenface/pca.py           80      0     30      0   100%
src/eigenface/qr.py            44      0     18      0   100%
src/eigenface/vector.py        30      0     14      0   100%
src/tests/test_matrix.py       46      0      0      0   100%
src/tests/test_pca.py          70      0      0      0   100%
src/tests/test_qr.py           31      0      6      0   100%
src/tests/test_vector.py       30      0      0      0   100%
-----------------------------------------------------------------------
TOTAL                         392      0    106      0   100%
```

# Mitä on testattu ja miten

Tällä hetkellä kaikki algoritmin keskeiset osat on testattu yksikkötesteillä sekä muutamalla laajemmalla integraatiotestillä.

## Vektorit

Testattu toiminnot: pistetulo, normi, normalisointi, vektorin vähennys ja skalaarilla jakaminen.
Syötteinä käytettiin sekä normaaleja vektoreita että reunatapauksia, kuten eri pituisia vektoreita ja nollalla jakoa.

## Matriisit

Testattu toiminnot: matriisin validointi, transpoosi, matriisikertolasku, matriisi–vektori-kertolasku sekä identiteettimatriisin luonti.
Syötteinä käytettiin normaaleja tapauksia sekä virhetilanteita (kelvoton matriisi ja väärän kokoiset syötteet).

## QR-dekompositio

QR-dekomposition oikeellisuus varmistettiin kokoamalla matriisi uudelleen ja tarkistamalla, että Q * R vastaa alkuperäistä matriisia.
Funktiolle `qr_decompose` syötettiin 2×2-matriisi, josta saatiin Q- ja R-matriisit. Tämän jälkeen nämä kerrottiin keskenään `matrix_multiplication`-funktiolla ja tulosta verrattiin alkuperäiseen matriisiin.

Lisäksi testattiin:

* Q-matriisin ortonormaalius
* R-matriisin yläkolmiomuoto
* eigendecompose diagonaalimatriisilla

## PCA ja tunnistus

Testattiin seuraavat toiminnot: keskiarvon laskeminen, datan keskitys, kovarianssimatriisi, `train_eigenfaces`, `predict_face`, euklidinen etäisyys ja nearest neighbor.
Syötteinä käytettiin normaaleja tapauksia sekä reunatapauksia (väärät koot ja nollanormi).

# Toteutus

Testit toteutettiin Pythonin pytest-kirjastolla. Jokaiselle moduulille tehtiin oma testitiedosto.

## Testien toisto

Testit voi toistaa luomalla virtuaaliympäristön esimerkiksi komennolla:

```
python3 -m venv venv
```

Tämän jälkeen asennetaan riippuvuudet:

```
poetry install
```

Testit suoritetaan komennolla:

```
python3 -m coverage run --branch -m pytest src
```

Ja raportti saadaan komennolla:

```
coverage report -m
```
