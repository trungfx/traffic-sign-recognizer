from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from predict import predict
import tempfile
# import json

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file'
                    )


class Hello(Resource):
    @staticmethod
    def get():
        return {'status': True}


class Image(Resource):
    @staticmethod
    def post():
        args = parser.parse_args()
        the_file = args['file']
        # save a temporary copy of the file
        ofile, ofname = tempfile.mkstemp()
        the_file.save(ofname)
        # predict
        results = predict(ofname)
        # formatting the results as a JSON-serializable structure:
        results_json = jsonify(results)
        print(results)

        return results_json


api.add_resource(Hello, '/hello')
api.add_resource(Image, '/image')

if __name__ == '__main__':
    app.run()
