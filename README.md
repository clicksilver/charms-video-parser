# Setup

* Install python: https://www.python.org/downloads/windows/

* Install opencv: `python -m pip install opencv`

* Install pytesseract: `python -m pip install pytesseract`

* Locate the install folder for tesseract.exe (this defaults to `C:\Program Files\Tesseract-OCR` but it could also be in `C:\Program Files (x86)\Tesseract-OCR`).

  * Edit the `pytesseract.pytesseract.tesseract_cmd` correspondingly in main.py

# Usage

From the command line: `python main.py PATH_TO_VIDEO`
