package com.example.ai_project;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    TextView pred_text;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button addPhotoButton = findViewById(R.id.add_button);
        pred_text = findViewById(R.id.pred_view);
        addPhotoButton.setOnClickListener(v -> openGallery());
    }

    private void openGallery() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Photo"), PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null) {
            try {
                InputStream imageStream = getContentResolver().openInputStream(data.getData());
                Bitmap imageBitmap = BitmapFactory.decodeStream(imageStream);
                if (imageBitmap != null) {
                    Bitmap resizedImage = Bitmap.createScaledBitmap(imageBitmap, 256, 256, false);
                    sendImageToServer(resizedImage);
                } else {
                    Log.e("MYLOG", "Bitmap could not be decoded");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void sendImageToServer(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();

        // Example API endpoint URL
        String apiUrl = "http://192.168.203.105:5000/predict";
        File imageFile = new File(getCacheDir(), "image.jpg");

        try {
            imageFile.createNewFile();
            FileOutputStream fos = new FileOutputStream(imageFile);
            fos.write(byteArray);
            fos.flush();
            fos.close();

            OkHttpClient client = new OkHttpClient();

            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("file", imageFile.getName(),
                            RequestBody.create(MediaType.parse("image/*"), imageFile))
                    .build();

            Request request = new Request.Builder()
                    .url(apiUrl)
                    .post(requestBody)
                    .build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    String responseData = response.body().string();
                    try {
                        JSONObject jsonObject = new JSONObject(responseData);
                        Map<String, Object> responseMap = new HashMap<>();
                        Iterator<String> keys = jsonObject.keys();
                        while (keys.hasNext()) {
                            String key = keys.next();
                            Object value = jsonObject.get(key);
                            responseMap.put(key, value);
                        }
                        runOnUiThread(() -> {
                            Object pred = responseMap.get("pred");
                            Log.d("MYLOG", "Response from server: " + responseData);
                            assert pred != null;
                            pred_text.setText(pred.toString());
                            pred_text.setVisibility(View.VISIBLE);
                        });
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }

                @Override
                public void onFailure(Call call, IOException e) {
                    Log.d("MYLOG", e.toString());
                    e.printStackTrace();
                }
            });

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
