# Integrare Unity - Python Face Verification

Acest proiect foloseste arhitectura descrisa in raport:

- Python ruleaza serverul de recunoastere faciala.
- Unity este clientul C# si simulatorul casei inteligente.
- Comunicarea se face prin socket TCP pe `127.0.0.1:5055`.

## 1. Pregateste baza de date

Porneste aplicatia existenta si inregistreaza cel putin o fata din PyCharm sau din terminal:

```powershell
python face_verification_app.py
```

Daca folosesti `uv` si il ai instalat in PATH, poti rula echivalent:

```powershell
uv run python face_verification_app.py
```

Foloseste camera sau o imagine, completeaza numele persoanei si apasa `Register Current Face`.

## 2. Porneste serverul pentru Unity

Din folderul proiectului Python:

```powershell
python unity_socket_server.py
```

In PyCharm poti crea o configuratie noua de Run pentru fisierul `unity_socket_server.py`.
Daca folosesti `uv`, comanda echivalenta este:

```powershell
uv run python unity_socket_server.py
```

Serverul accepta comenzile:

- `PING`
- `STATUS`
- `RELOAD`
- `VERIFY`

Raspunsurile sunt JSON pe o singura linie.

## 3. Adauga scripturile in Unity

Copiaza fisierele din folderul `UnityScripts` in proiectul Unity, in:

```text
Assets/Scripts/
```

Fisiere:

- `FaceAuthClient.cs`
- `DoorAccessController.cs`
- `SmartRoomEnergyController.cs`

## 4. Configureaza scena Unity

1. Creeaza un `Player` cu controller first-person si tag `Player`.
2. Creeaza un empty object numit `FaceAuthClient`.
3. Adauga scriptul `FaceAuthClient` pe acel object.
4. Lasa `host = 127.0.0.1` si `port = 5055`.
5. Pentru usa principala, creeaza un obiect/pivot care se poate roti.
6. Creeaza in fata usii un cub transparent numit `DoorAccessTrigger`.
7. Pe trigger bifeaza `Box Collider > Is Trigger`.
8. Adauga scriptul `DoorAccessController` pe trigger.
9. Leaga in Inspector:
   - `Auth Client` -> objectul `FaceAuthClient`
   - `Door Pivot` -> pivotul usii
   - optional `Status Text` -> textul UI pentru feedback

## 5. Configureaza luminile inteligente

Pentru fiecare camera:

1. Creeaza un cub invizibil care acopera camera.
2. Adauga `Box Collider` si bifeaza `Is Trigger`.
3. Adauga scriptul `SmartRoomEnergyController`.
4. In `Controlled Lights`, adauga luminile camerei.
5. Seteaza `Watts Per Light`, de exemplu `60`.
6. Optional leaga textele UI pentru status si consum.

## 6. Fluxul demonstratiei

1. Pornesti `unity_socket_server.py`.
2. Pornesti scena Unity.
3. Te apropii de usa.
4. Unity trimite `VERIFY` la Python.
5. Python captureaza imaginea de la camera, verifica liveness si compara fata cu baza de date.
6. Daca raspunsul este `VERIFIED`, Unity deschide usa.
7. La intrarea in camera, luminile se aprind automat.
8. La iesirea din camera, luminile se sting si consumul virtual ramane afisat.

## Observatie pentru raport

Integrarea finala demonstreaza separarea responsabilitatilor:

- Python realizeaza procesarea biometrica si validarea autenticitatii.
- Unity interpreteaza rezultatul si actioneaza componentele casei inteligente.
- Socket-ul TCP permite comunicare locala, rapida si fara servicii cloud.
