import unittest
import os
from appFixed import app  # Import your Flask app here

class FlaskAppTest(unittest.TestCase):
    def setUp(self):
        # Setup a test client
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Testing the /predict endpoint
        with open('room_pic.png', 'rb') as img:
            response = self.app.post(
                '/predict',
                content_type='multipart/form-data',
                data={'file': img}
            )
            self.assertEqual(response.status_code, 200)
            # You can add more assertions here to check the content of the response

if __name__ == '__main__':
    unittest.main()