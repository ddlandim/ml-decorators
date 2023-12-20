import os
import requests
import json


def main(event, context):
    # Get the video URI from the event
    video_uri = event['name']

    # Make a request to the inference model
    response = requests.post('https://example.com/inference', data={'video_uri': video_uri})

    # Parse the response
    inference_result = json.loads(response.content)

    # Move the video to the appropriate folder
    if inference_result['detected']:
        destination_folder = 'detected'
    else:
        destination_folder = 'not_detected'
    os.rename(video_uri, os.path.join('gs://my-bucket', destination_folder, os.path.basename(video_uri)))

    # Publish an event to the Pub/Sub topic
    if inference_result['detected']:
        topic = 'my-pubsub-topic'
        data = json.dumps({'camera_id': os.path.basename(video_uri.split('/')[5])})
        requests.post('https://pubsub.googleapis.com/v1/projects/my-project/topics/{}/publish'.format(topic),
                       data=data)


if __name__ == '__main__':
    main(None, None)