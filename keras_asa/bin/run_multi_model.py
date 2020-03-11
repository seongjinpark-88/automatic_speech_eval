# run the code created in multi_input_model.py

import multi_input_model as modeler

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/S07/wav", "~/opensmile-2.3.0", "../../SJP_JC_Audio/output")
acoustic_features = phonetic_test.get_features(supra=False)
