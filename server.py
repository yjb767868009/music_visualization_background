import datetime
import os
from flask import *
from music_recognition import Music_Recognition

app = Flask(__name__)

music_rec = Music_Recognition()


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
    app.run('172.16.13.43', port=8081)
