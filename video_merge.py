###list_videos contains the path the the videos=[]
import subprocess as sp
import os

async def merge_videos(list_video):
    width = 1920
    height = 1080
    input_videos = ""
    input_setpts = "nullsrc=size={}x{} [base];".format(width, height)
    input_overlays = "[base][video0] overlay=shortest=1 [tmp0];"
    grid_width = 4
    grid_height = 4
    for index, path_video in enumerate(list_video):
            input_videos += " -i " + path_video
            input_setpts += "[{}:v] setpts=PTS-STARTPTS, scale={}x{} [video{}];".format(index, width//grid_width, height//grid_height, index)
            if index > 0 and index < len(list_video) - 1:
                input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={} [tmp{}];".format(index-1, index, width//grid_width * (index%grid_width), height//grid_height * (index//grid_width), index)
            if index == len(list_video) - 1:
                input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={}".format(index-1, index, width//grid_width * (index%grid_width), height//grid_height * (index//grid_width))

    complete_command = "ffmpeg" + input_videos + " -filter_complex \"" + input_setpts + input_overlays + "\" -c:v libx264 output.mp4"

    output = os.popen(complete_command)
    os.rename("output.mp4","./merge_videos/output.mp4")