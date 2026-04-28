## Käyttöohjeet

1. Lataa zip-tiedosto Gitistä  
2. Pura zip-tiedosto haluamaasi kansioon ja siirry sen sisälle  
3. Asenna riippuvuudet:

```bash
poetry install
```

---

## Data

Ohjelma olettaa koulutusdatan seuraavanlaista kansiorakennetta:

```text
dataset/
  person1/
    1.pgm
    2.pgm
    ...
  person2/
    1.pgm
    2.pgm
    ...
```

### Tuetut kuvaformaatit

- `.pgm`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

Valmista dataa löytyy esimerkiksi:  
https://cam-orl.co.uk/facedatabase.html

Kun data on ladattu, sijoita se kansioon:

```text
src/eigenface/data
```

---

## Mallin koulutus

Kouluta malli komennolla:

```bash
poetry run python -m eigenface.cli train
```

Näet kaikki parametrit komennolla:

```bash
poetry run python -m eigenface.cli -h
```

### Parametrit

- `--k` (default: `20`)  
  Valitaan top 20 suurimman ominaisarvon komponenttia.

- `--threshold` (default: `9.0`)  
  Tunnistuskynnys. Mitä suurempi arvo sitä löysempi tunnistus.

- `--dataset` (default: `"src/eigenface/data"`)  
  Koulutusdatan sijainti.

- `--size` (default: `64`)  
  Kuvien koko (esim. 64x64).

- `--iterations` (default: `10`)  
  QR-dekomposition iteraatiot. Mitä suurempi sitä tarkempi mutta hitaampi.

- `--tolerance` (default: `1e-4`)  
  Iterointi pysähtyy, kun muutos on tätä pienempi.

- `--model-out` (default: `"eigenface/models/eigenface_model.json"`)  
  Tallennettavan mallin sijainti.

---

## Mallin testaus

Aja ennustus komennolla:

```bash
poetry run python -m eigenface.cli predict <image_path>
```

Lisäparametrit:

```bash
poetry run python -m eigenface.cli predict -h
```

### Parametrit

- `image` *(pakollinen)*  
  Polku testikuvaan.

- `--model` (default: `"eigenface/models/eigenface_model.json"`)  
  Käytettävän mallin polku.



