# Interactive Vision-Assisted Object Grasping for the Visually Impaired

## 1. Project Overview

This project is an assistive vision-language system designed to help a visually impaired user locate and grasp a target drink. The system combines real-time camera input, speech interaction, vision-language inference, and spoken feedback. A user first speaks the object they want to pick up, such as a bottle of cola or Sprite. The client application captures the latest camera frame and sends it together with a concise task prompt to the backend server. The server runs the fine-tuned model and returns a short action-oriented instruction, such as "move left", "move forward", or "grab now". The application then reads this instruction aloud and continues the process in a loop until the user stops the guidance session.

From a system perspective, the repository is organized into three main parts: `app`, `server`, and `trainer`. The `app` module is responsible for user interaction and device-side media processing. The `server` module provides a deployable inference service for the model. The `trainer` module contains the scripts used for data preparation, model tuning, and offline inference. In the current implementation, the end-to-end user experience mainly depends on the coordination between the app and the server.

The overall workflow of the project can be summarized as follows:

```text
User speaks target object + camera captures scene
        |
        v
App module
  - speech recognition
  - frame acquisition
  - prompt construction
  - HTTP request
        |
        v
Server module
  - model loading
  - image decoding
  - SmolVLM + LoRA inference
  - JSON response
        |
        v
App module
  - display result
  - speak guidance aloud
```

This design turns a trained model into a usable assistive pipeline instead of only an offline experiment.

## 2. App Module

The `app` module is the interactive front end of the project and is responsible for turning the model into a usable assistive tool. Instead of treating inference as an offline experiment, the app provides a real-time workflow in which the user speaks a target object, the device captures the surrounding scene, and the system returns spoken step-by-step guidance. This module is therefore not only a visualization layer, but also the component that manages user interaction, device permissions, media input, networking, and speech output.

The app is implemented with SwiftUI, and its control center is `app/FastVLM App/ContentView.swift`. This file coordinates the major runtime objects of the application, including `CameraController`, `RemoteVLMModel`, `SpeechRecognizer`, and `SpeechPlayer`. It also stores the key interface states used during interaction, such as the latest frame, the current transcript, whether guidance is active, and the currently displayed output. From an implementation point of view, `ContentView` acts as a lightweight state machine for the whole assistive loop.

### 2.1 Startup and Preparation Logic

One important part of the app is the startup stage. The view launches several asynchronous tasks when it appears. These tasks perform the following actions:

- configure a shared audio session for simultaneous recording and playback,
- start the camera immediately,
- request speech-recognition and microphone permissions,
- call `model.load()` to test server connectivity,
- warm up the remote model wrapper and speech recognizer,
- start distributing video frames from the camera stream into the UI.

This preparation flow is important because the first interaction in a multimodal app is often the least stable. If the application waits until the user presses the talk button before checking permissions or backend availability, the user experiences a larger delay and failures become harder to diagnose. In the current implementation, the app completes these checks in advance and keeps a `preparationComplete` flag so that the interface can disable interaction until the required components are ready.

The audio session is configured with `AVAudioSession` in `.playAndRecord` mode and uses spoken-audio settings. This choice is necessary because the app needs to do two things in the same session: record the user's speech and later play back synthesized guidance. If these states are not managed carefully, audio interruptions can easily occur.

### 2.2 Speech Interaction Design

Speech input is implemented in `SpeechRecognizer.swift`. The recognizer defines a simple internal state machine with four states: `idle`, `preparing`, `recording`, and `finalizing`. This state split is useful because speech capture on Apple platforms is not instantaneous. The recognizer must first prepare the audio engine, then record audio, then leave a short time window for final transcription before fully stopping. By making these phases explicit, the app can avoid starting a new recording too early or cancelling a valid partial result.

The push-to-talk interaction in `ContentView.swift` is implemented with a `DragGesture(minimumDistance: 0)`. The gesture starts recording on the first touch event and stops recording when the finger is released. This design avoids using a tap gesture because a tap is too short for natural speech input. The code also keeps a `pushToTalkArmed` flag so that the recognizer is started only once per press and is not retriggered multiple times while the gesture is still active.

When recording stops, the app does not process the command immediately. Instead, `handleUserCommand()` waits briefly before reading the transcript. This delay is small, but it is important because the recognizer may still be in the finalizing phase and may append additional words or punctuation. After the final transcript is available, the app stores it in `userGoalSpeech` and extracts a cleaner object name for display in the UI. The object extraction logic removes common language prefixes such as "I want to pick up", "grab", or "help me get" so that the target label shown on screen is shorter and easier to read.

### 2.3 Camera Pipeline and Frame Management

The camera subsystem is implemented mainly in `app/Video/CameraController.swift`. This class wraps `AVCaptureSession`, discovers available camera devices, chooses an appropriate input device, and sends captured sample buffers into an asynchronous stream. The app prefers a wide field of view where available, including ultra-wide cameras on iOS, because the grasping task depends on seeing both the user's hand and the target object within the same frame.

The captured frames are passed through `AsyncStream<CMSampleBuffer>` and then transformed into an `AsyncStream<CVImageBuffer>` for display and inference. An important implementation detail is the use of `.bufferingNewest(1)`. This means the app intentionally keeps only the newest frame in the stream rather than queueing many historical frames. This is a critical design decision for real-time guidance, because stale visual input is less useful than slightly lower frame continuity. By dropping old frames, the system avoids accumulating lag during periods of slower inference or speech playback.

`VideoFrameView.swift` renders the current frame in the interface while `ContentView` separately stores the same frame in `latestFrame` for inference. This separation allows the preview and the backend request logic to stay synchronized without tightly coupling display code and inference code. On iOS, the camera pipeline also updates the rotation angle dynamically using `AVCaptureDevice.RotationCoordinator`, ensuring that the image orientation remains consistent as the device moves.

### 2.4 Continuous Guidance Loop

The most important runtime behavior of the app is the continuous guidance loop. This logic is implemented in `startGuidanceLoop()` inside `ContentView.swift`. Once the user has spoken a command, the app creates a `Task` and repeatedly executes the following sequence:

1. read the latest available frame,
2. construct a concise prompt,
3. cancel any stale model state if needed,
4. call remote inference,
5. receive the short instruction,
6. speak the instruction aloud,
7. wait until playback finishes,
8. start the next iteration.

This loop continues until the user explicitly stops guidance, at which point `stopGuidance()` cancels the task, resets the model state, and stops any ongoing speech output.

The loop is intentionally serial. The app does not send the next request while the previous spoken instruction is still being read aloud. This is one of the most important design choices in the whole project. In a normal vision app, parallel inference might improve throughput. However, in an assistive setting, overlapping commands can confuse the user and degrade usability. The serial design therefore sacrifices some raw responsiveness in exchange for clarity and interaction stability.

### 2.5 Prompt Construction and Response Constraints

The prompt is generated by `ConcisePromptTemplate.swift`. The template is fixed except for the user-goal line, which is inserted after sanitation. Newlines are removed, quotation marks are normalized, and the text is trimmed before inclusion. The prompt tells the model that it is guiding a blind user to grasp the requested drink and that it should reply with one short spoken command only.

This prompt design is important because the project is not a general image-captioning system. The model is expected to return short action commands, not long scene descriptions. By constraining the app-side prompt to a narrow format, the system keeps inference behavior aligned with the training objective and reduces the chance of receiving verbose or inconsistent output.

### 2.6 Remote Inference Client

`RemoteVLMModel.swift` is the app-side networking layer. It is annotated with `@Observable` and `@MainActor`, which simplifies UI updates and ensures that state changes remain synchronized with the SwiftUI view hierarchy. The class exposes state such as `running`, `output`, `promptTime`, and `evaluationState`, making it easy for the interface to show whether the model is idle, preparing a request, or generating a response.

The `load()` method performs a `GET /health` request to confirm that the configured server is reachable. This method also interprets the JSON response and sets a human-readable `modelInfo` string for debugging and status reporting. The current implementation keeps the `warmup()` method for API compatibility, even though inference is remote rather than on-device.

The `generate()` method contains the full client-side inference flow. It first prevents overlapping requests by checking the `running` flag, then switches the state to `processingPrompt`, converts the `CVPixelBuffer` to JPEG, and finally submits the HTTP request. JPEG conversion is implemented with `CoreImage`, `CGImage`, and `CGImageDestination`, which is an efficient path on Apple platforms and keeps the image format standardized before network transfer.

The HTTP request itself is built manually as `multipart/form-data`. This means the app explicitly appends the prompt field, the `max_new_tokens` field, and the image file with the correct boundary markers. Building the request this way gives full control over what the server receives and avoids introducing unnecessary third-party networking code. After the request returns, the client decodes the JSON payload into an internal `InferJSONResponse` struct and places the returned command into `output`.

Another small but useful implementation detail is latency measurement. `generate()` records a start timestamp and stores a `promptTime` string in milliseconds when the response arrives. This makes it possible to observe rough end-to-end request cost from the app side during testing.

### 2.7 Audio Playback and Synchronization

The final stage of the app-side pipeline is speech synthesis, implemented in `SpeechPlayer.swift`. This class wraps `AVSpeechSynthesizer` and exposes two important methods: `speak()` and `waitUntilDone()`. The `speak()` method stops any current utterance before starting a new one, preventing overlapping speech. The `waitUntilDone()` method bridges the speech synthesizer delegate callbacks into an async continuation, which allows the guidance loop to pause until playback is complete.

This synchronization between speech output and the guidance loop is a key implementation detail. Without it, the app would continue requesting new instructions while the previous one was still being spoken, which would make the interaction much harder to follow. By serializing inference and playback together, the app provides a more coherent assistive experience.

Overall, the `app` module is responsible not only for user interaction, but also for temporal coordination across all real-time components. It synchronizes camera frames, speech input, network requests, and speech output into a single usable loop. This is the reason why it is one of the most important parts of the whole system.

## 3. Server Module

The `server` module is the deployable inference layer of the project. Its role is to keep the trained model loaded in memory, expose a simple HTTP API, and convert app-side requests into multimodal inference results. This module is implemented with FastAPI in `server/main.py`, and it is designed as a lightweight but structured service rather than a one-off experiment script. In the overall system, the server is the bridge between the mobile client and the model produced by the training pipeline.

### 3.1 Service Interface and API Design

The backend exposes two endpoints. `GET /health` returns a small JSON object indicating that the service is alive. This endpoint is intentionally simple because it is mainly used by the app during startup to verify basic connectivity. `POST /infer` is the main endpoint used during interaction. It accepts a `multipart/form-data` request with three fields:

- `prompt`, which contains the text prompt constructed by the app,
- `image`, which contains the current frame encoded as an image file,
- `max_new_tokens`, which controls the maximum length of generation.

The response is returned in JSON format with a cleaned `text` field and a `raw` field containing the full decoded generation. Returning both forms is useful during development. The app normally uses only the cleaned text, but the raw output is helpful for debugging prompt formatting and model behavior.

This API design is deliberately narrow. The backend does not attempt to support general multi-user chat, streaming tokens, or many different model tasks. Instead, it focuses on exactly what the project needs: a stable image-plus-prompt inference call with a concise response.

### 3.2 Startup Path and Configuration

The server can be launched either with `python -m server.main` or through `uvicorn`. Startup parameters include the base model path, LoRA adapter path, host, port, and an optional device override. Internally, the entry point parses command-line arguments, normalizes local paths to absolute paths, places the final values into environment variables, and then starts the FastAPI app. The repository also includes `server/__main__.py`, which allows `python -m server` to work as a short convenience command.

One useful implementation detail is `_normalize_model_path()`. This helper keeps Hugging Face model identifiers unchanged but converts local model directories into absolute paths. This avoids failures caused by changing the current working directory when launching the service. Since model loading can already be fragile, reducing path-related ambiguity is valuable in practice.

### 3.3 Model Lifecycle Management

The most important server-side implementation choice is that the model is loaded once during application startup rather than once per request. This is handled through FastAPI's lifespan function. During startup, the server reads `SMOL_MODEL_PATH`, `SMOL_ADAPTER_PATH`, and the optional `SMOL_DEVICE`, then calls `load_model_and_processor()` from `trainer/infer.py`. The returned model, processor, and resolved device are stored in a process-level dictionary named `_state`.

This design has several advantages. First, it avoids the extremely high cost of repeatedly loading a multimodal model and adapter on every request. Second, it keeps inference latency low enough for interactive use. Third, it makes the server architecture closer to a real deployment service, where model initialization is expensive but inference calls are frequent. On shutdown, the lifespan function clears the stored state and, when CUDA is available, explicitly frees cached GPU memory through `torch.cuda.empty_cache()`.

### 3.4 Request Handling Flow

The `/infer` endpoint implements a clear request-processing pipeline. When the app sends a request, the server performs the following steps:

1. validate the `max_new_tokens` range,
2. read the uploaded image bytes,
3. reject empty uploads,
4. decode the image with Pillow and convert it to RGB,
5. retrieve the loaded model, processor, and device from `_state`,
6. call the shared inference function,
7. return a JSON response with `text` and `raw`.

The route also maps failures to explicit HTTP status codes. Invalid parameters or unreadable images return `400`, missing model state returns `503`, and unexpected inference failures return `500`. This makes the API significantly easier to integrate with because the client can distinguish between bad requests, startup problems, and actual runtime inference errors.

The decision to convert every uploaded image to RGB before inference is also important. It ensures that different image formats from the client side are normalized into a consistent form before passing them into the model processor. This reduces variability in input handling and simplifies debugging.

### 3.5 Shared Inference Logic with the Trainer Module

Rather than duplicating model logic inside the server, the backend reuses the shared functions in `trainer/infer.py`. This file contains several core helpers:

- `pick_device()`, which chooses CUDA, MPS, or CPU,
- `pick_dtype()`, which selects the tensor precision appropriate to that device,
- `load_model_and_processor()`, which loads the processor, base model, and optional PEFT adapter,
- `infer_one()`, which runs a single image-plus-prompt inference pass,
- `extract_assistant()`, which cleans the generated output.

This reuse is important for consistency. The same processor and the same conversation formatting logic are used both in standalone inference and in the deployed server. `infer_one()` constructs a message in the expected chat format, inserts an image token and the text prompt, applies the processor's chat template, and then calls `processor()` with both the image and text. After moving the tensors to the chosen device, the function runs `model.generate()` and decodes the result. Finally, `extract_assistant()` trims away the prompt scaffolding so that only the assistant's reply is returned to the client.

By structuring the server around these shared functions, the project reduces the risk of mismatches between offline inference and online serving. This is especially important for a vision-language model, where even small changes in prompt wrapping or preprocessing can alter behavior significantly.

### 3.6 Device and Precision Strategy

A useful server-side detail is the way hardware selection is handled. If no device is forced by the server configuration, the backend automatically prefers CUDA when available, then MPS, and finally CPU. The tensor precision is also adapted to the selected backend:

- CUDA uses `bfloat16`,
- MPS uses `float16`,
- CPU uses `float32`.

This logic is not very long in code, but it is important because it makes the server portable across different development machines. The same repository can be used on a GPU workstation, an Apple Silicon laptop, or a CPU-only environment without rewriting the service code. From a project perspective, this flexibility made development and testing much easier.

### 3.7 Non-Streaming Response Design

The current server returns a complete JSON response only after generation finishes. It does not stream partial tokens. This was a deliberate design choice. A streaming interface would be more complex to implement on both the server and app sides, especially because the app is designed to speak complete short commands rather than partial text fragments. In the current task setting, a single compact response is simpler, more reliable, and easier to integrate with the client's serial guidance loop.

### 3.8 Testing and Developer Support

The `server` module also includes developer-oriented support files. `server/API.md` documents the service address, required environment variables, endpoint behavior, form fields, and example requests in both cURL and Python. `server/test_api.py` provides an integration test script that first checks `GET /health` and then, unless disabled, sends an image to `POST /infer`.

This script includes useful options such as `--health-only`, a configurable timeout, a custom base URL, and a custom prompt. These details are valuable during development because they allow the backend to be validated independently from the mobile client. When debugging an end-to-end system, the ability to isolate backend correctness is extremely important.

In summary, the `server` module is more than a thin wrapper around the model. It handles startup configuration, resource management, request validation, model reuse, and integration consistency. It is the part of the project that makes the trained model callable in real time and therefore makes the app-side assistive interaction possible.

## 4. Trainer Module
### 4.1 Data Preprocessing and Dataset Construction

The training data used in this project is built from raw images collected for a beverage-grabbing scenario. Because the original images may come from different devices and may use different file formats, the first step is to standardise them before automatic labeling and model training.

To standardise the image input, the preprocessing pipeline converts supported formats such as PNG, WebP, BMP, TIFF, HEIC, and HEIF into `.jpg`. Files with the `.jpeg` suffix are renamed to `.jpg` directly. Images are converted to RGB and saved as JPEG with a fixed quality setting.

After image standardisation, the dataset is generated automatically rather than labeled fully by hand. A multimodal teacher model is used to examine each image and produce task-specific supervision. In the current pipeline, each image is paired with several beverage targets, including `sprite`, `cola`, and `lemon_tea`. For each target, the teacher model produces a short command-style answer that can later be used as supervision for fine-tuning.

The generated samples are stored in a file-based structure. Each sample uses a stem of the form `<base>__<drink_key>__vNN`, where the middle segment records the target beverage class and the final segment marks the prompt variant. Under this naming scheme, one sample is typically stored as an image file together with a prompt file and an answer file, for example `<stem>.jpg`, `<stem>.prompt.txt`, and `<stem>.ans.txt`. The pipeline can also save intermediate reasoning text, but the final training stage mainly uses the answer file as the supervision target.

This file-based structure also makes the dataset easier to inspect and debug. Because the class information is encoded in the file name, the later training stage can recover the target category without relying on an extra metadata table. In addition, multiple prompt variants can be generated for the same image and class combination, so the dataset can be expanded without requiring a new round of image collection.

### 4.2 Prompt Design

Prompt design is an important part of the training pipeline because it affects both the quality of the automatically generated labels and the language diversity seen during fine-tuning. In this project, two prompt styles are used for different purposes: a full instruction prompt for teacher-model labeling, and a concise prompt format for student-model training.

For automatic dataset generation, the full prompt gives the teacher model a clear task definition. It frames the problem as blind-assistance guidance and asks the model to output only short spoken-style commands instead of long descriptions. The prompt also restricts the output space to task-relevant responses such as `move left`, `move forward`, `grab now`, `show your hand`, or `object missing`. This keeps the generated labels close to the form required in the final application.

For fine-tuning, a more compact prompt format is used. Instead of relying on one fixed wording, the concise prompt module defines several aliases for each drink category and combines them with multiple pickup-intent templates. For example, the cola target may appear as `cola`, `a cola`, `the cola can`, or `the red-label cola`, while the user request can be phrased as “I want to pick up {name}.”, “Please help me grab {name}.”, or similar alternatives. This makes the language input more varied while keeping the task unchanged.

The prompt system supports both deterministic and random generation. Since each dataset stem already contains the drink key, the target class can be recovered directly from the file name. A deterministic mode can generate a stable prompt for a given stem, while the training pipeline can also sample aliases and templates dynamically. As a result, the same image may appear with slightly different user-goal wording across epochs. This reduces reliance on a single phrasing pattern and makes the model less sensitive to wording changes.

Overall, the prompt design does not change the visual content of the task. Instead, it increases linguistic variation around the same target object, which is helpful for a system expected to respond to naturally phrased user requests.

### 4.3 Model Selection and Fine-tuning Method

The core of our vision-language reasoning engine is based on the **SmolVLM-256M-Instruct** architecture. This model is a compact yet powerful Vision-Language Model (VLM) that integrates a lightweight vision encoder with a small-scale language model, making it ideal for the latency-sensitive requirements of assistive grasping tasks.

To achieve high-performance adaptation while minimizing the computational footprint, we implement a comprehensive fine-tuning strategy:

* **Model Initialization and Device Mapping**: The system implements an automated hardware detection logic via `pick_device()` and `pick_dtype()`. It dynamically assigns the model to `CUDA` (using `bfloat16`), `MPS` (using `float16`), or `CPU` (using `float32`). This ensures optimal floating-point precision and memory utilization across different training environments.
* **Low-Rank Adaptation (LoRA) Configuration**: Rather than updating all 256M parameters, we apply **LoRA (Low-Rank Adaptation)** to specific linear projections. According to the `build_lora_config()` implementation, we target the following modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. We set the rank ($r$) to 8 and the scaling factor ($\alpha$) to 16, utilizing a dropout rate of 0.05 to prevent overfitting on the specialized grasping dataset.
* **Quantization Strategy (QLoRA)**: For hardware with limited VRAM, the training pipeline supports **4-bit quantization** through the `BitsAndBytesConfig`. By utilizing the `nf4` (NormalFloat 4) data type and double quantization, the base model weights are frozen in a compressed state, while only the high-precision LoRA adapters remain trainable. The model is further prepared for k-bit training using `prepare_model_for_kbit_training()` to ensure gradient stability.

### 4.4 Training Data Organization and Training Objective

Once the dataset has been prepared, each sample must be converted into a format suitable for multimodal instruction tuning. The training stage treats every sample as a paired image–instruction example in which the model receives an image and a user request, and is expected to produce a short guidance command as the response.

The training pipeline supports two prompt modes. In the default concise mode, the target beverage class is parsed directly from the sample stem, and the user-side prompt is generated during training. In the full mode, the prompt is read from the stored prompt file. This design allows the same dataset to support both fixed prompts and dynamically generated prompts, depending on the training setup.

During batch construction, each sample is organised as a short dialogue. The user message contains the image token together with the text prompt, and the assistant message contains the expected answer. These messages are then formatted into the chat-style input required by SmolVLM. In this way, the model sees training data in the same general form that it will later encounter during inference.

The supervision objective is defined only on the assistant response. Tokens that belong to the user prompt are masked out during loss computation, and padding tokens are excluded as well. The special image token is also removed from the loss. Therefore, the optimisation process does not reward the model for repeating the prompt or reproducing formatting tokens. It only rewards the model for generating the target command itself.

This is consistent with the task setting. In the final application, the model is not required to describe the whole image or restate the user request. Its job is to produce the next short action instruction, such as `move left`, `move up`, or `grab now`. By applying the loss only to the assistant side of the sequence, the training objective remains aligned with this deployment goal.

The sample construction workflow is shown below.

```text
raw image + answer file (+ optional full prompt file)
        ↓
recover target class from file name or read stored prompt
        ↓
build user message with image + prompt
        ↓
append assistant answer
        ↓
format sample into chat-style multimodal input
        ↓
tokenize and batch samples
        ↓
mask prompt tokens, padding tokens, and image token
        ↓
compute loss only on assistant answer
```

### 4.5 Training Workflow and Implementation Details

The training execution is managed by a custom-configured `Trainer` pipeline, which translates the raw dataset into a deployable instruction-following model. The workflow consists of the following technical stages:

* **Execution Arguments**: The system utilizes a sophisticated `TrainingArguments` setup in `train.py`. Key parameters include:
    * **Optimization**: We employ the `adamw_torch` optimizer with a weight decay of 0.01 and a linear warmup ratio of 0.05.
    * **Learning Rate Schedule**: A conservative learning rate of $1 \times 10^{-4}$ is used to ensure the pre-trained multimodal representations are not distorted during adaptation.
    * **Throughput Management**: To simulate a stable global batch size on limited hardware, we combine a `per_device_train_batch_size` (default 1) with `gradient_accumulation_steps` (default 4).
* **Memory Efficiency Techniques**: To mitigate the high VRAM consumption of multimodal gradients, we enable **Gradient Checkpointing** (`gradient_checkpointing=True`). This reduces memory overhead by recomputing intermediate activations during the backward pass. Additionally, `dataloader_pin_memory` is strategically toggled based on the device type to optimize data transfer speeds between CPU and GPU.
* **Logging and Checkpointing Strategy**: 
    * **Monitoring**: The pipeline integrates **TensorBoard** via the `report_to="tensorboard"` argument. It logs training loss and optimization metrics every 5 steps, providing granular visibility into the convergence process.
    * **Persistence**: We implement an epoch-based saving strategy (`save_strategy="epoch"`) with a `save_total_limit=2`. This ensures that the system automatically retains the most recent checkpoints while discarding older ones to manage storage efficiency.
* **Post-Training Artifacts**: Upon completion of `trainer.train()`, the system explicitly saves the final LoRA adapters and the associated `AutoProcessor`. This ensures that the specific tokenization and image rescaling logic used during training are perfectly preserved for the inference server.

### 4.6 Inference Testing and Performance Evaluation

The table below is reserved for future evaluation results.

| Test Item | Description | Value |
|-----------|-------------|-------|
| Model | Fine-tuned SmolVLM-256M | TBD |
| Task | Beverage grasping guidance | TBD |
| Average latency (ms) | Mean inference time over repeated runs | TBD |
| Minimum latency (ms) | Best-case inference time | TBD |
| Maximum latency (ms) | Worst-case inference time | TBD |
| Std. deviation (ms) | Latency variation across runs | TBD |
| Number of runs | Total benchmark repetitions | TBD |
| Hardware | Testing device / platform | TBD |

### 4.7 Summary

The `trainer` module provides a robust and scalable infrastructure for specialized VLM adaptation. By leveraging **SmolVLM-256M-Instruct** as a foundation and applying **PEFT/LoRA** techniques, the system bridges the gap between general-purpose vision models and domain-specific assistive tools. 

The implementation effectively addresses three critical challenges: (1) **Data Efficiency**, through the use of target-aware prompt generation; (2) **Computational Efficiency**, through QLoRA and gradient checkpointing; and (3) **Deployment Readiness**, by outputting standardized adapter weights compatible with the real-time inference server. This training architecture ensures that the final model can provide accurate, low-latency spatial guidance for visually impaired users in diverse household environments.

## 5. Conclusion
