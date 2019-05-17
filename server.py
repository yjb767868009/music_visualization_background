import datetime
import os
from flask import *
from music_recognition import Music_Recognition
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
music_rec = Music_Recognition()


@app.after_request
def af_request(resp):
    """
    #请求钩子，在所有的请求发生后执行，加入headers。
    :param resp:
    :return:
    """
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


@app.route('/music', methods=['POST'])
def upload():
    upload_file = request.files['file']
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if upload_file:
        file_path = os.path.join('upload', str(now_time) + '.wav')
        upload_file.save(file_path)
        beat_times, db_list, style = music_rec.analysis(file_path)
        print('success save to %s' % file_path)
        return jsonify({'message': 'success',
                        'beat_times': beat_times,
                        'db_list': db_list,
                        'style': style,
                        })
    return jsonify({'message': 'fail'})


if __name__ == '__main__':
    app.run('0.0.0.0', port=8081)
