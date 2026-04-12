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
