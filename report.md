# Interactive Vision-Assisted Object Grasping for the Visually Impaired

## 1. Project Overview

This project is an assistive vision-language system designed to help a visually impaired user locate and grasp a target drink. The system combines real-time camera input, speech interaction, vision-language inference, and spoken feedback. A user first speaks the object they want to pick up, such as a bottle of cola or Sprite. The client application captures the latest camera frame and sends it together with a concise task prompt to the backend server. The server runs the fine-tuned model and returns a short action-oriented instruction, such as "move left", "move forward", or "grab now". The application then reads this instruction aloud and continues the process in a loop until the user stops the guidance session.

From a system perspective, the repository is organized into three main parts: `app`, `server`, and `trainer`. The `app` module is responsible for user interaction and device-side media processing. The `server` module provides a deployable inference service for the model. The `trainer` module contains the scripts used for data preparation, model tuning, and offline inference. In the current implementation, the end-to-end user experience mainly depends on the coordination between the app and the server.


## 2. App Module

The `app` module is the interactive front end of the project and is responsible for turning the model into a usable assistive tool. Instead of treating inference as an offline experiment, the app provides a real-time workflow in which the user speaks a target object, the device captures the surrounding scene, and the system returns spoken step-by-step guidance. This module is therefore not only a visualization layer, but also the component that manages user interaction, device permissions, media input, networking, and speech output.

The app is implemented with SwiftUI, and its control center is `app/FastVLM App/ContentView.swift`. This file coordinates the major runtime objects of the application, including `CameraController`, `RemoteVLMModel`, `SpeechRecognizer`, and `SpeechPlayer`. It also stores the key interface states used during interaction, such as the latest frame, the current transcript, whether guidance is active, and the currently displayed output. From an implementation point of view, `ContentView` acts as a lightweight state machine for the whole assistive loop.

### 2.1 Startup and Preparation Logic

On launch, the view asynchronously configures the `AVAudioSession` in `.playAndRecord` mode, starts the camera, requests speech and microphone permissions, verifies server connectivity via `model.load()`, and begins distributing video frames to the UI. All checks complete before a `preparationComplete` flag is set, keeping the interface disabled until every component is ready. This upfront initialization avoids delays and diagnostic difficulties that would arise if permissions or backend availability were only checked at the moment of first interaction.

### 2.2 Speech Interaction Design

Speech input is implemented in `SpeechRecognizer.swift`, which defines a four-state machine (`idle`, `preparing`, `recording`, `finalizing`). These explicit phases prevent the app from starting a new recording before the previous transcription has fully completed.

The push-to-talk interaction uses a `DragGesture(minimumDistance: 0)` in `ContentView.swift`: recording starts on touch-down and stops on release. A `pushToTalkArmed` flag ensures the recognizer is triggered only once per press. After release, `handleUserCommand()` waits briefly for the finalizing phase to finish, then stores the transcript in `userGoalSpeech` and strips common prefixes such as "I want to pick up" to extract a cleaner target label for display.

### 2.3 Camera Pipeline and Frame Management

The camera subsystem is implemented in `CameraController.swift`, which wraps `AVCaptureSession` and preferentially selects a wide-angle or ultra-wide camera so that both the user's hand and the target object remain visible in the same frame.

Captured frames flow through `AsyncStream<CMSampleBuffer>` → `AsyncStream<CVImageBuffer>` with `.bufferingNewest(1)`, meaning only the latest frame is retained. This avoids accumulating lag during slower inference or speech playback. `VideoFrameView.swift` renders the preview, while `ContentView` independently stores `latestFrame` for inference, keeping display and model logic decoupled. On iOS, `AVCaptureDevice.RotationCoordinator` dynamically corrects orientation as the device moves.

### 2.4 Continuous Guidance Loop

The core runtime behavior is the continuous guidance loop in `startGuidanceLoop()`. After the user speaks a command, the app repeatedly reads the latest frame, constructs a prompt, calls remote inference, speaks the returned instruction aloud, and waits until playback finishes before starting the next iteration. The user can stop guidance at any time via `stopGuidance()`, which cancels the task and resets all state.

The loop is intentionally serial: the next request is never sent while the previous instruction is still being spoken. This sacrifices some throughput but prevents overlapping commands from confusing the user, which is critical in an assistive context.

### 2.5 Prompt Construction and Response Constraints

The prompt is generated by `ConcisePromptTemplate.swift`. The template is fixed except for the user-goal line, which is inserted after sanitation. Newlines are removed, quotation marks are normalized, and the text is trimmed before inclusion. The prompt tells the model that it is guiding a blind user to grasp the requested drink and that it should reply with one short spoken command only.

### 2.6 Remote Inference Client

`RemoteVLMModel.swift` is the app-side networking layer, annotated with `@Observable` and `@MainActor` to keep UI state synchronized. It exposes properties such as `running`, `output`, and `promptTime` so the interface can reflect the current inference status.

The `load()` method sends a `GET /health` request to verify server connectivity. The `generate()` method guards against overlapping calls via a `running` flag, converts the current `CVPixelBuffer` to JPEG through `CoreImage`, and submits a hand-built `multipart/form-data` request containing the prompt, `max_new_tokens`, and the image. The server response is decoded into an `InferJSONResponse` struct whose `text` field is stored in `output`. A start-to-finish timestamp is also recorded as `promptTime` for latency profiling.

### 2.7 Audio Playback and Synchronization

The final stage of the app-side pipeline is speech synthesis, implemented in `SpeechPlayer.swift`. This class wraps `AVSpeechSynthesizer` and exposes two important methods: `speak()` and `waitUntilDone()`. The `speak()` method stops any current utterance before starting a new one, preventing overlapping speech. The `waitUntilDone()` method bridges the speech synthesizer delegate callbacks into an async continuation, which allows the guidance loop to pause until playback is complete.

This synchronization between speech output and the guidance loop is a key implementation detail. Without it, the app would continue requesting new instructions while the previous one was still being spoken, which would make the interaction much harder to follow. By serializing inference and playback together, the app provides a more coherent assistive experience.

## 3. Server Module

The `server` module is the deployable inference layer of the project. Its role is to keep the trained model loaded in memory, expose a simple HTTP API, and convert app-side requests into multimodal inference results. This module is implemented with FastAPI in `server/main.py`, and it is designed as a lightweight but structured service rather than a one-off experiment script. In the overall system, the server is the bridge between the mobile client and the model produced by the training pipeline.

### 3.1 Service Interface and API Design

The backend exposes two endpoints. `GET /health` returns a small JSON object indicating that the service is alive. This endpoint is intentionally simple because it is mainly used by the app during startup to verify basic connectivity. `POST /infer` is the main endpoint used during interaction. It accepts a `multipart/form-data` request with three fields:

- `prompt`, which contains the text prompt constructed by the app,
- `image`, which contains the current frame encoded as an image file,
- `max_new_tokens`, which controls the maximum length of generation.

The response is returned in JSON format with a cleaned `text` field and a `raw` field containing the full decoded generation. Returning both forms is useful during development. The app normally uses only the cleaned text, but the raw output is helpful for debugging prompt formatting and model behavior.

### 3.2 Startup Path and Configuration

The server can be launched either with `python -m server.main` or through `uvicorn`. Startup parameters include the base model path, LoRA adapter path, host, port, and an optional device override. Internally, the entry point parses command-line arguments, normalizes local paths to absolute paths, places the final values into environment variables, and then starts the FastAPI app. The repository also includes `server/__main__.py`, which allows `python -m server` to work as a short convenience command.

One useful implementation detail is `_normalize_model_path()`. This helper keeps Hugging Face model identifiers unchanged but converts local model directories into absolute paths. This avoids failures caused by changing the current working directory when launching the service. Since model loading can already be fragile, reducing path-related ambiguity is valuable in practice.

### 3.3 Model Lifecycle Management

The most important server-side implementation choice is that the model is loaded once during application startup rather than once per request. This is handled through FastAPI's lifespan function. During startup, the server reads `SMOL_MODEL_PATH`, `SMOL_ADAPTER_PATH`, and the optional `SMOL_DEVICE`, then calls `load_model_and_processor()` from `trainer/infer.py`. The returned model, processor, and resolved device are stored in a process-level dictionary named `_state`.

This design has several advantages. First, it avoids the extremely high cost of repeatedly loading a multimodal model and adapter on every request. Second, it keeps inference latency low enough for interactive use. Third, it makes the server architecture closer to a real deployment service, where model initialization is expensive but inference calls are frequent. On shutdown, the lifespan function clears the stored state and, when CUDA is available, explicitly frees cached GPU memory through `torch.cuda.empty_cache()`.

### 3.4 Request Handling Flow

The `/infer` endpoint validates the `max_new_tokens` range, reads and decodes the uploaded image to RGB via Pillow, retrieves the preloaded model and processor from `_state`, runs the shared inference function, and returns a JSON response containing both a cleaned `text` field and a `raw` field. Failures are mapped to explicit HTTP status codes (`400` for bad input, `503` for missing model state, `500` for runtime errors), allowing the client to distinguish between different error categories.

### 3.5 Shared Inference Logic with the Trainer Module

Rather than duplicating model logic, the server reuses shared functions from `trainer/infer.py`, including `pick_device()` / `pick_dtype()` for hardware selection, `load_model_and_processor()` for loading the base model with an optional LoRA adapter, and `infer_one()` for running a single image-plus-prompt inference pass. `infer_one()` formats the input as a chat-style message, applies the processor's template, runs `model.generate()`, and returns the decoded result after `extract_assistant()` strips away the prompt scaffolding. Sharing this code path between offline evaluation and the deployed server ensures consistent preprocessing and prompt formatting.

The current server returns a complete JSON response only after generation finishes. It does not stream partial tokens. This was a deliberate design choice. A streaming interface would be more complex to implement on both the server and app sides, especially because the app is designed to speak complete short commands rather than partial text fragments. In the current task setting, a single compact response is simpler, more reliable, and easier to integrate with the client's serial guidance loop.


## 4. Trainer Module

### 4.1 Data Preprocessing and Dataset Construction

The training dataset is constructed from raw images collected during the beverage-grabbing task. Since the original data may use different image formats, the preprocessing stage first standardises them before labeling and training.

In the current pipeline, supported formats such as PNG, WebP, BMP, TIFF, HEIC, and HEIF are converted into `.jpg`. Files with the `.jpeg` suffix are renamed to `.jpg` directly. During this process, images are converted to RGB and saved as JPEG with a fixed quality setting.

After preprocessing, the dataset is generated automatically rather than labeled fully by hand. A multimodal teacher model is used to examine each image and produce short command-style supervision for the target beverage classes, including `sprite`, `cola`, and `lemon_tea`.

Each sample is stored with a structured stem of the form `<base>__<drink_key>__vNN`. A typical sample contains an image file, a prompt file, and an answer file, such as `<stem>.jpg`, `<stem>.prompt.txt`, and `<stem>.ans.txt`. Intermediate reasoning text may also be saved, but the final training stage mainly uses the answer file. Because the class information is encoded in the file name, the later training pipeline can recover the target category directly.

### 4.2 Prompt Design

Prompt design is used in both dataset generation and fine-tuning. In this project, two prompt styles are used: a full instruction prompt for teacher-model labeling, and a concise prompt format for student-model training.

For automatic labeling, the full prompt defines the task as blind-assistance guidance and restricts the output to short spoken-style commands such as `move left`, `move forward`, `grab now`, `show your hand`, or `object missing`. This keeps the generated labels close to the output format required by the final system.

For fine-tuning, a concise prompt mechanism is used. Each drink category has multiple aliases, and these are combined with different pickup-intent templates. For example, the cola target may appear as `cola`, `a cola`, `the cola can`, or `the red-label cola`, while the user request may be written in different short forms. The prompt can be generated either deterministically from the sample stem or randomly during training. This makes the language input more varied and reduces dependence on a single phrasing pattern.

### 4.3 Model Selection and Fine-tuning Method

The model used in this project is **SmolVLM-256M-Instruct**. It is a lightweight vision-language model, which is suitable for this task because the system needs to run with limited computing resources and respond quickly.

For adaptation, the training pipeline uses **LoRA** instead of full-parameter fine-tuning. In the training script, LoRA is applied to key projection layers in the transformer, including attention layers and MLP layers. The pipeline also supports 4-bit quantization on compatible CUDA hardware, which helps reduce memory usage during training.

![Figure 4.1. Overall pipeline of the model and application workflow.](figure4_1.png)

*Figure 4.1. Overall pipeline of the model and application workflow.*

### 4.4 Training Data Organization and Training Objective

During training, each sample is treated as a paired image–instruction example. The model receives an image together with a user request and is trained to generate a short guidance command as the answer.

The training pipeline supports two prompt modes. In the default concise mode, the drink class is parsed directly from the sample stem and the prompt is generated during training. In the full mode, the prompt is read from the stored prompt file. This allows the same dataset to support both stored prompts and dynamically generated prompts.

During batch construction, each sample is organised as a short dialogue. The user side contains the image token and the text prompt, while the assistant side contains the expected answer. These messages are then converted into the chat-style input required by SmolVLM.

The loss is applied only to the assistant response. Tokens from the user prompt are masked out during loss computation, and padding tokens and the image token are excluded as well. As a result, the model is trained to generate the target command itself rather than repeat the prompt.

![Figure 4.2. Teacher–student training idea used in the project.](figure4_2.png)

*Figure 4.2. Teacher–student training idea used in the project.*

### 4.5 Training Workflow and Implementation Details

Training is implemented with the Hugging Face `Trainer` pipeline. The script sets the main training arguments, including batch size, learning rate, gradient accumulation, and checkpoint saving. To reduce memory usage, the training process supports gradient checkpointing, and TensorBoard is used for logging.

After training, the LoRA adapter and the corresponding processor are saved for later inference. This keeps the training and deployment settings consistent.

![Figure 4.3. LoRA training result and loss curve.](figure4_3.png)

*Figure 4.3. LoRA training result and loss curve.*

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

The trainer module covers dataset preparation, prompt construction, and model fine-tuning for the beverage-grabbing task. It connects raw image data, automatically generated supervision, and the final model used by the later inference system.


## 5. Conclusion
