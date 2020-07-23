package com.example.model_tuner;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.Context;
import android.content.Intent;
import android.location.LocationManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;
import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.RECORD_AUDIO;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static java.lang.Thread.sleep;

public class MainActivity extends AppCompatActivity {
    int helpDetectCount = 0;
    int inferenceCount = 0;
    int sampleNumber = 0;
    double avgNoise = 0;
    boolean isAutoThreshold=false;
    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 1000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);


    private static final String LOG_TAG = "AudioRecordTest";
    private static final String OUTPUT_SCORES_NAME = "output0";
    private static final String INPUT_DATA_NAME = "input_1_1";
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    boolean shouldContinueRecognition = true;
    boolean shouldContinue = true;
    short[] recordingBuffer = new short[RECORDING_LENGTH];
    int recordingOffset = 0;
    private Thread recordingThread;
    private Thread recognitionThread;
    public static double threshold = 0.95;
    private static int recognitionResultLength = 5;

    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILENAME = "file:///android_asset/model71.pb";
    private boolean[] recognitionResult = new boolean[recognitionResultLength];
    final float score[] = new float[recognitionResultLength];

    ArrayAdapter<String> adapter;
    ArrayList<String> array= new ArrayList<>();
    TextView count;

    public static String[] PERMISSIONS =
            new String[]{ACCESS_FINE_LOCATION, CAMERA,
                    WRITE_EXTERNAL_STORAGE, RECORD_AUDIO};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);
        requestAllPermissions();
        setContentView(R.layout.activity_main);
//        final View view = getLayoutInflater().inflate(R.layout.list_item,null,false);

        ListView list= findViewById(R.id.list1);
        Button setThreshold=findViewById(R.id.btn);
        final EditText getThreshold=findViewById(R.id.editThreshold);
        setThreshold.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                double th = threshold;
                try {
                    th= new Float(getThreshold.getText().toString());
                }catch (Exception e){
                    System.out.println(e.getMessage());
                }
                if (th<=1 && th>=0){
                    threshold=th;
                }
            }
        });
        count=findViewById(R.id.helpCount);
//        adapter=new ArrayAdapter<String>(this,android.R.layout.simple_list_item_1,array);
        adapter=new ArrayAdapter<String>(this,R.layout.list_item,array);
        list.setAdapter(adapter);

        final Switch autoThreshold = findViewById(R.id.switch2);
        autoThreshold.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if (b){
                    isAutoThreshold=true;
                }else{
                    isAutoThreshold=false;
                }
            }
        });
        Switch toggleSwitch =findViewById(R.id.switch1);
        toggleSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    Toast.makeText(MainActivity.this, "Recording", Toast.LENGTH_SHORT).show();
                    startRecording();
                    startRecognition();

                } else {
                    stopRecording();
                    stopRecognition();
                    helpDetectCount=0;
                    array.clear();
                    adapter.notifyDataSetChanged();


                }
            }
        });

    }
    public void requestAllPermissions() {
        ActivityCompat.requestPermissions(this, PERMISSIONS, 1);
    }
    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        //  Log.v(LOG_TAG,"");
        // Estimate the buffer size we'll need for this device.
        int bufferSize =
                AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }
        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.DEFAULT,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            Log.v(LOG_TAG, "Recording state: " + record.getState());
            return;
        }

        record.startRecording();

        Log.v(LOG_TAG, "Start recording");

        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;

            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock();
            if(secondCopyLength>0 || newRecordingOffset==maxLength)sampleNumber++;
            try {
                System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
                recordingOffset = newRecordingOffset % maxLength;
            }
            catch (Exception e){
                System.out.println(e.getMessage());
            }finally{
//                System.out.println("Audio buff len "+audioBuffer.length+" record buff length "+recordingBuffer.length);
                recordingBufferLock.unlock();
            }
        }

        record.stop();
        record.release();
    }

    private void recognize() {
        Log.v(LOG_TAG, "Start recognition");
        short[] inputBuffer = new short[RECORDING_LENGTH];
        double[] doubleInputBuffer = new double[RECORDING_LENGTH];
        final float[] outputScores = new float[1];
        String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};
        int localSampleNumber=0;
        double cummAvgNoise=0;
        int cummAvg=0;
        int avg=0;
        int temp = 0;
        boolean newSample=false;

        while (shouldContinueRecognition) {
            inferenceCount++;
            recordingBufferLock.lock();
            if(localSampleNumber<sampleNumber){
                newSample=true;
                localSampleNumber=sampleNumber;
            }else{
                newSample=false;
            }
            try {
                int maxLength = recordingBuffer.length;
                System.arraycopy(recordingBuffer, 0, inputBuffer, 0, maxLength);
            } finally {
                recordingBufferLock.unlock();
            }
            // We need to feed in float values between -1.0 and 1.0, so divide the signed 16-bit inputs.
            for (int i = 0; i < RECORDING_LENGTH; ++i) {
                doubleInputBuffer[i] = inputBuffer[i] / 32767.0;
            }
            //MFCC java library.
            MFCC mfccConvert = new MFCC();
            float[] mfccInput = mfccConvert.process(doubleInputBuffer);

            final double[] sum={0};
            for(double f : doubleInputBuffer){
                f=f*100; if(f<0) f=0-f; sum[0]=sum[0]+f;
            }
            avg++;
            sum[0] = sum[0]/doubleInputBuffer.length;
            System.out.println("Scores: "+Arrays.toString(score)+"\n Noise: "+sum[0]);
//            System.out.println("localSampleNo: "+localSampleNumber+" sampleNumber: "+sampleNumber);

            if(newSample){
//                System.out.println("localSampleNo: "+localSampleNumber+" sampleNumber: "+sampleNumber);
//                localSampleNumber=sampleNumber;
                avgNoise= sum[0];
                cummAvg++;
                cummAvgNoise=cummAvgNoise+avgNoise;
                if(sampleNumber%5==0){
                    cummAvgNoise=cummAvgNoise/cummAvg;
                    if(isAutoThreshold){
                        if(cummAvgNoise<.20){
                            threshold=.80;
                        }else if(cummAvgNoise<.50){
                            threshold=.88;
                        }else if(cummAvgNoise<1){
                            threshold=.90;
                        }else if(cummAvgNoise<2){
                            threshold=.93;
                        }else if(cummAvgNoise<3){
                            threshold=.95;
                        }else if(cummAvgNoise<7){
                            threshold=.97;
                        }else{
                            threshold=.98;
                        }
                    }
                    avg=0;
                    cummAvg=0;
                    cummAvgNoise=0;
                }
            }
            //System.out.println("input buff len "+doubleInputBuffer.length+" mfcc buff length "+mfccInput.length);
            // Run the model.
            inferenceInterface.feed(INPUT_DATA_NAME, mfccInput, 1, 126, 40);
            inferenceInterface.run(outputScoresNames);
            inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);

            final boolean isRecognised = outputScores[0] > threshold;

            for (int i = 0; i < recognitionResultLength - 1; i++) {
//                recognitionResult[i] = recognitionResult[i + 1];
                score[i] = score[i + 1];
            }
//            recognitionResult[recognitionResultLength - 1] = isRecognised;
            score[recognitionResultLength - 1] = outputScores[0];
            final boolean[] isPreviousRecognised1 = {true};
            if (isRecognised) {
//                int recogNo=0;
//                for (boolean r: recognitionResult){
//                    if (r) recogNo++;
//                }if(recogNo==4){
//                    if(true){
//                        isPreviousRecognised1[0]=false;
//                        helpDetectCount++;
//                        onDetectingHelp();
//                    }
//                }
                boolean isPreviousRecognised = false;
                for (boolean i : recognitionResult) {
                    isPreviousRecognised |= i;
                }
                isPreviousRecognised1[0]=isPreviousRecognised;
                if (isPreviousRecognised) {
                } else {
                    helpDetectCount++;
                    onDetectingHelp();
                }
            }
            for (int i = 0; i < recognitionResultLength - 1; i++) {
                recognitionResult[i] = recognitionResult[i + 1];
//                score[i] = score[i + 1];
            }recognitionResult[recognitionResultLength - 1] = isRecognised;



            temp = inferenceCount % recognitionResultLength;
            if (isRecognised&& !isPreviousRecognised1[0] || temp==0) {
//                Log.v(LOG_TAG, inferenceCount + " MFCC Input====> " + Arrays.toString(doubleInputBuffer));
//                Log.v(LOG_TAG, inferenceCount + " MFCC Output====> " + Arrays.toString(mfccInput));
                Handler handler = new Handler(Looper.getMainLooper());
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        String[] arr=new String[5];
                        for(int i=0;i<arr.length;i++){
                            arr[i]=String.format("%.3f", score[i]);
                        }
                        if(isRecognised&& !isPreviousRecognised1[0]) array.add("HELP DETECTED :\n"+Arrays.toString(arr)+" noise: "+String.format("%.2f",sum[0])+ " threshold: "+String.format("%.3f",threshold));
                        else array.add(Arrays.toString(arr)+ " noise:"+String.format("%.2f",sum[0])+ " threshold:"+String.format("%.2f",threshold));
                        adapter.notifyDataSetChanged();
                        count.setText("Help Detected: "+helpDetectCount+"\nNoise: "+String.format("%.2f", sum[0])+"\nScore: "+String.format("%.3f",outputScores[0])+"\nAvg Noise: "
                                +String.format("%.2f", avgNoise)+"\nThreshold: "+String.format("%.2f",threshold)+"\nSample Number: "+sampleNumber);
                    }
                });
            }
        }
    }

    public synchronized void startRecording() {
        if (recordingThread != null) {
            return;
        }
        shouldContinue = true;
        recordingThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                record();
                            }
                        });
        recordingThread.start();
    }

    public synchronized void startRecognition() {
        if (recognitionThread != null) {
            return;
        }
        shouldContinueRecognition = true;
        recognitionThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                recognize();
                            }
                        });
        recognitionThread.start();
    }

    public synchronized void stopRecording() {
        avgNoise=0;
        sampleNumber=0;
        inferenceCount=0;
        if (recordingThread == null) {
            return;
        }
        shouldContinue = false;
        recordingThread = null;
    }

    public synchronized void stopRecognition() {
        avgNoise=0;
        sampleNumber=0;
        inferenceCount=0;
        if (recognitionThread == null) {
            return;
        }
        shouldContinueRecognition = false;
        recognitionThread = null;
    }
    public void onDetectingHelp() {
        Log.d("TAG", "onDetectingHelp: Detected Help");
        Handler handler = new Handler(Looper.getMainLooper());
        handler.post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(getApplicationContext(),
                        "Help Detected",
                        Toast.LENGTH_SHORT).show();
            }
        });
//        Toast.makeText(this, "Help Detected", Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onDestroy() {
        stopRecording();
        stopRecognition();
        super.onDestroy();
    }
}
