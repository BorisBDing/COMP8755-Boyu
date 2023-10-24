Project readme:

Note: You also can see the User Guidline video and artefact performance under this link (https://anu365-my.sharepoint.com/:f:/g/personal/u7350592_anu_edu_au/EgwmTXzDdN1Nnrc47RGdPzgBjozcPpx2P-fWkUDauPdGgg?e=t5BBia). This link also includ all code about my project.

The target user for this project: this artefact is design for Data scientists and researchers who are interested in bias research.

This project utilise different multi-modal models and click models to study the bias in existing multi-modal searches, and quantitatively analyzes the bias of different models between different keywords.

User Guidline:
1. Required Environment:

jina                    3.20.0
streamlit               1.25.0
regex                   2023.6.3
torchvision             0.15.2
docarray                0.21.1
open-clip-torch         2.20.0
transformers            4.31.0
ftfy                    6.1.1
torch                   2.0.1
numpy                   1.24.2
pandas                  2.0.3
Pillow                  9.5.0
pillow-avif-plugin      1.3.1
annlite                 0.5.10

2. Project Instruction:

a. This project can configure the CLIPEncoder model in myclip/server/torch-flow.yml, and the corresponding model needs to be downloaded.
b. The data set is configured in the format of index, century, country, objid, image, medium, category.
c. In order to simplify the operation process, this multi-modal search uses text to search for images in the database.

3. Who to use this atifact：
a. Start the server: python3 __main__.py

b. Visualization of search: 
streamlit run frontend.py

c. Evaluate bias:
python3 evaluate.py

4. Outcome of the evaluation：
key：search the required keywords
key object attribute-name content-bias interface-bias
