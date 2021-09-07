from flask import Flask, jsonify, abort, request
from flask_restplus import Api, Resource
import plugins
import os
import json
import pickle
from core.utils import load_dataset, load_model_class

def get_plugins():
    models = os.listdir("plugins")
    models = filter(lambda x: not x.startswith("__"), models)
    return map(lambda x: x.split('.')[0], models)

flask_app = Flask(__name__)
api = Api(app = flask_app)


print("Loading model and pipeline")
with open("./config.json") as f:
    config = json.load(f)

model_classes = {}
for p in config["packages"]:
    if p['attribute'] is None:
        DataPipeline = load_dataset(p['name'])
    else:
        model_classes[p['name']] = load_model_class(p['name'])

model_dir = "./trained_models"

models = []
for model_file in sorted(os.listdir(model_dir)):
    models.append(model_classes[config['model_type']].load(os.path.join(model_dir, model_file)))

with open('./pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

headers = pipeline.get_input_headers()

plugins_dic = {}
for plugin_name in get_plugins():
    if hasattr(plugins, plugin_name):
        plugin = getattr(plugins, plugin_name)
        if hasattr(plugin, "serve"):
            plugins_dic[plugin_name] = getattr(plugin, "serve")

def plugin(plugin):
    if plugin in plugins_dic:
        try:
            result = plugins_dic[plugin](request.values, models, pipeline)
        except ValueError as e:
            print(e)
            return abort(422)
        return jsonify(result)
    abort(404)

print("Setting up plugin routes")
name_space = api.namespace('plugin', description='Main APIs')

@api.route('/plugin/<plugin>', endpoint='plugin')
@api.doc(params={'plugin': 'The plugin you want to query'})
class PluginResource(Resource):
    def get(self, plugin):
        return plugin(plugin)

    def post(self, plugin):
        return plugin(plugin)


@api.route('/model', endpoint='model')
@api.doc(params={})
class PluginResource(Resource):
    def get(self, plugin):
        return plugin(plugin)

    def post(self, plugin):
        return plugin(plugin)

if __name__ == "__main__":
    flask_app.run(debug=True)