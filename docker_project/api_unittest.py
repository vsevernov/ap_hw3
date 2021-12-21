import unittest
import requests

class FlaskApiTest(unittest.TestCase):
    def test_get_model_info(self):
        name = 'test_model'
        r1 = requests.get(f'http://127.0.0.1:5000/api/ml_models/model_info/{name}')
        self.assertEquals(r1.status_code, 200)

    def test_get_models_info(self):
        r2 = requests.get(f'http://127.0.0.1:5000/api/ml_models/models_info')
        self.assertEquals(r2.status_code, 200)


    def test_update(self):
        old_name = 'test_model'
        new_name = 'test_model_v1'
        input_data = {'old_name': old_name, 'new_name': new_name}
        r3 = requests.post('http://127.0.0.1:5000/api/ml_models/update', json=input_data)
        self.assertEquals(r3.json()['info'], 'success update')


    def test_get_delete(self):
        name = '7_rfc_model'
        r4 = requests.get(f'http://127.0.0.1:5000/api/ml_models/model_info/{name}')
        self.assertEquals(r4.status_code, 200)
        r4 = requests.delete(f'http://127.0.0.1:5000/api/ml_models/predict/{name}')
        self.assertEquals(r4.status_code, 200)
        r4 = requests.get(f'http://127.0.0.1:5000/api/ml_models/model_info/{name}')
        self.assertEquals(r4.status_code, 500)

if __name__ == "__main__":
    unittest.main()


