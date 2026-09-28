/**
 * VSLR REALTIME COMMUNICATION STUDIO - WEB APPLICATION
 * Đại học Cần Thơ (CTU) — NCKH VSLR 2026
 * 
 * Tích hợp Full-stack với Backend Python:
 * - Luồng hình ảnh MJPEG tốc độ cao 30+ FPS bám sát cử chỉ bàn tay (MediaPipe Holistic C++)
 * - Suy luận mô hình PyTorch BiLSTM v3 (24 nhãn chuẩn VSLR) qua Server-Sent Events (SSE)
 * - Bộ phát âm giọng đọc chuẩn VieNeu-TTS (Giọng Trúc Ly 48kHz)
 * - Điều khiển chuyển đổi camera thiết bị (HD Webcam / Camera A16)
 */

document.addEventListener('DOMContentLoaded', () => {

  // =========================================================================
  // 1. GESTURE DATA DICTIONARY (24 VSLR Standard Gestures)
  // =========================================================================
  const GESTURE_DATA = [
    { id: 'xin_chao', name: 'Xin chào', cat: 'Giao tiếp', icon: '👋', desc: 'Bàn tay mở, lòng bàn tay hướng về phía trước, vẫy nhẹ từ trái qua phải ngang tầm mắt.', tip: 'Giữ cánh tay góc 90 độ, vẫy 2 nhịp dứt khoát.' },
    { id: 'cam_on', name: 'Cảm ơn', cat: 'Giao tiếp', icon: '🙏', desc: 'Chạm các đầu ngón tay phải vào cằm hoặc môi dưới, sau đó đưa thẳng tay hướng ra phía trước đối phương.', tip: 'Chuyển động dứt khoát hướng về trước, kết hợp gật đầu nhẹ.' },
    { id: 'tam_biet', name: 'Tạm biệt', cat: 'Giao tiếp', icon: '🙋', desc: 'Đưa bàn tay lên ngang vai, các ngón tay khép mở nhịp nhàng hoặc vẫy ngang chào tạm biệt.', tip: 'Thực hiện trong khoảng 1-2 giây.' },
    { id: 'xin_loi', name: 'Xin lỗi', cat: 'Giao tiếp', icon: '🙇', desc: 'Nắm hờ bàn tay phải, xoay tròn nhẹ trên vùng ngực trái kèm nét mặt hối lỗi.', tip: 'Duy trì tư thế ngực ổn định để landmarks phần thân chuẩn xác.' },
    { id: 'lau_roi_khong_gap', name: 'Lâu rồi không gặp', cat: 'Giao tiếp', icon: '⏳', desc: 'Vuốt nhẹ má xuống từ cằm, sau đó đưa hai tay mở ra phía trước.', tip: 'Cử chỉ kết hợp 2 pha: thời gian "Lâu" và "Gặp".' },
    { id: 'rat_vui_duoc_gap_ban', name: 'Rất vui được gặp bạn', cat: 'Giao tiếp', icon: '🤝', desc: 'Đặt hai lòng bàn tay vuốt ngực nhẹ hướng lên, sau đó đưa 2 ngón trỏ lại gần nhau.', tip: 'Cử chỉ 2 tay đối xứng (Bimanual gesture).' },
    { id: 've_nha_can_than', name: 'Về nhà cẩn thận', cat: 'Giao tiếp', icon: '🏠', desc: 'Hai bàn tay tạo hình mái nhà, sau đó đưa hai tay sang hai bên gập mở cảnh báo cẩn thận.', tip: 'Đỉnh mái nhà giữ ngang tầm mắt khoảng 0.3s.' },
    { id: 'hom_nay_ban_khoe_khong', name: 'Hôm nay bạn khỏe không', cat: 'Sức khỏe', icon: '🩺', desc: 'Chỉ tay về phía trước, sau đó nắm hai tay siết nhẹ kéo xuống và hơi nghiêng đầu hỏi.', tip: 'Siết cơ tay dứt khoát để MediaPipe ghi nhận gia tốc khuỷu tay.' },
    { id: 'toi_khoe', name: 'Tôi khỏe', cat: 'Sức khỏe', icon: '💪', desc: 'Chỉ ngón trỏ vào ngực mình, sau đó hai tay nắm chặt đưa lên ngang ngực.', tip: 'Giữ tư thế nắm tay trong khoảng 10-15 frames cuối.' },
    { id: 'toi_binh_thuong', name: 'Tôi bình thường', cat: 'Sức khỏe', icon: '😐', desc: 'Bàn tay phải để sấp ngang bụng, lắc nhẹ bàn tay qua lại sang hai bên.', tip: 'Biên độ lắc tay vừa phải, giữ khoảng cách cơ thể 25-30cm.' },
    { id: 'toi_khong_khoe', name: 'Tôi không khỏe', cat: 'Sức khỏe', icon: '🤒', desc: 'Đặt một tay lên trán hoặc ngực, bàn tay còn lại vẫy lắc ngang thể hiện sự phủ định.', tip: 'Nét mặt hơi mệt mỏi hỗ trợ nhận diện chuẩn xác hơn.' },
    { id: 'ban_co_van_de_gi_khong', name: 'Bạn có vấn đề gì không', cat: 'Sức khỏe', icon: '❓', desc: 'Hai bàn tay xòe ngửa trước ngực, nhấp nhô luân phiên kèm ánh mắt quan tâm.', tip: 'Bàn tay giữ trong khung hình camera.' },
    { id: 'goi_xe_cuu_thuong', name: 'Gọi xe cứu thương', cat: 'Sức khỏe', icon: '🚑', desc: 'Một tay làm hình điện thoại áp tai, tay kia xoay vòng trên đỉnh đầu mô phỏng đèn cứu thương.', tip: 'Cử chỉ ưu tiên phát âm giọng nói khẩn cấp.' },
    { id: 'ban_ten_gi', name: 'Bạn tên gì', cat: 'Thông tin', icon: '🏷️', desc: 'Chỉ tay về phía đối diện, bắt chéo hai ngón trỏ và ngón giữa (Tên) và ngửa tay hỏi (Gì).', tip: 'Thực hiện thẳng vào chuyển động, tránh vẫy tay mở đầu.' },
    { id: 'ban_que_o_dau', name: 'Bạn quê ở đâu', cat: 'Thông tin', icon: '📍', desc: 'Chỉ tay đối diện, hai bàn tay khum nhẹ úp xuống rồi mở ngửa ra kèm cử chỉ hỏi Ở đâu.', tip: 'Vị trí tay tập trung ở khoảng giữa ngực và bụng.' },
    { id: 'may_tuoi', name: 'Mấy tuổi', cat: 'Thông tin', icon: '🎂', desc: 'Đưa tay vuốt nhẹ từ cằm xuống rồi bung các ngón tay rung nhẹ hỏi số lượng.', tip: 'Chuyển động các ngón tay cần dẻo và rõ ràng.' },
    { id: 'ban_dang_lam_gi', name: 'Bạn đang làm gì', cat: 'Thông tin', icon: '💼', desc: 'Hai tay nắm nhẹ gõ vào nhau hai lần (Làm việc) rồi xòe ngửa hai tay hỏi (Gì).', tip: 'Khoảng cách tiếp xúc giữa hai nắm tay cần chuẩn xác.' },
    { id: 'di_dau', name: 'Đi đâu', cat: 'Thông tin', icon: '🚶', desc: 'Ngón trỏ và ngón giữa chúc xuống bước đi, sau đó ngửa tay xoay nhẹ hỏi phương hướng.', tip: 'Chuyển động bước chân mô phỏng thực hiện trước ngực.' },
    { id: 'sieu_thi', name: 'Siêu thị', cat: 'Thông tin', icon: '🛒', desc: 'Hai tay nắm hờ đẩy về phía trước như đang đẩy xe mua hàng, kết hợp chọn đồ.', tip: 'Cử chỉ hai tay song song, giữ thẳng cẳng tay khi đẩy.' },
    { id: 'ban_co_can_giup_do_khong', name: 'Bạn có cần giúp đỡ không', cat: 'Tương tác', icon: '🤝', desc: 'Một tay ngửa, tay kia nắm đặt lên mu/lòng bàn tay dưới và cùng nâng nhẹ lên (Giúp đỡ).', tip: 'Hai tay liên kết tạo cụm điểm landmarks đặc trưng.' },
    { id: 'duoc_khong', name: 'Được không', cat: 'Tương tác', icon: '👌', desc: 'Bàn tay làm dấu Like hoặc chạm ngón cái vào ngón trỏ (OK), gật nhẹ cổ tay 2 lần.', tip: 'Cử chỉ ngắn gọn; giữ ổn định trong 0.8s.' },
    { id: 'nhu_the_nao', name: 'Như thế nào', cat: 'Tương tác', icon: '🤔', desc: 'Hai bàn tay úp xuống, sau đó lật ngửa lên đồng thời hai bên với nét mặt thắc mắc.', tip: 'Động tác lật bàn tay (Hand Flip) là mốc đặc trưng.' },
    { id: 'sao_the', name: 'Sao thế', cat: 'Tương tác', icon: '🤷', desc: 'Hai bàn tay mở rộng ngửa trước ngực, hơi nhún nhẹ vai và lắc cổ tay hỏi lý do.', tip: 'Tránh tạo hình tay chữ OK để không bị phân loại sai.' },
    { id: 'chuyen_gi', name: 'Chuyện gì', cat: 'Tương tác', icon: '💬', desc: 'Đưa hai ngón trỏ chỉ vào nhau rồi gõ nhẹ hai lần trước ngực kết hợp ngửa bàn tay hỏi.', tip: 'Chuyển động dứt khoát, không vung tay quá rộng.' }
  ].map((item, idx) => {
    const signers = ['P01', 'P02', 'P03', 'P04'];
    const signer = signers[idx % 4];
    return {
      ...item,
      signer,
      video: `tutorials/${item.id}.mp4?v=3.2`,
      thumb: `tutorials/${item.id}.jpg?v=3.2`
    };
  });

  // =========================================================================
  // 2. APPLICATION STATE & BACKEND ENDPOINT RESOLUTION
  // =========================================================================
  const urlParams = new URLSearchParams(window.location.search);
  const queryBackend = urlParams.get('backend');
  if (queryBackend) {
    localStorage.setItem('vslr_backend_url', queryBackend.replace(/\/+$/, ''));
  }

  const isLocalHost = (
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname.endsWith('.hf.space') ||
    window.location.hostname.endsWith('.trycloudflare.com')
  );
  const isDirectTunnel = window.location.hostname.endsWith('.trycloudflare.com');
  const isLocalPC = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const DEFAULT_REMOTE_BACKEND = 'https://ntbii305-vslr-backend.hf.space';
  const API_BASE = isLocalHost ? '' : (localStorage.getItem('vslr_backend_url') || DEFAULT_REMOTE_BACKEND);

  const state = {
    activeView: 'view-home',
    dictViewMode: 'cards', // 'cards' | 'videos'
    currentSpeed: 1.0,
    cameraEnabled: false,
    clientCamMode: !isLocalPC,
    clientStream: null,
    clientFrameTimer: null,
    signerName: localStorage.getItem('vslr_signer_name') || '',
    sentence: [],
    recMode: 'auto',       // 'auto' | 'manual'
    showHands: true,
    isRecording: false,
    ttsVoice: 'Trúc Ly',
    currentCamera: 'HD Webcam',
    backendConnected: false,
    eventSource: null
  };

  // DOM Elements
  const heroPredictedWord = document.getElementById('heroPredictedWord');
  const predConfVal = document.getElementById('predConfVal');
  const recStatusTag = document.getElementById('recStatusTag');
  const recStatusText = document.getElementById('recStatusText');
  const btnMainToggleTranslate = document.getElementById('btnMainToggleTranslate');
  const sentenceWordsBox = document.getElementById('sentenceWordsBox');
  const liveStreamImg = document.getElementById('liveStreamImg');
  const liveVideo = document.getElementById('liveVideo');
  const camStandbyScreen = document.getElementById('camStandbyScreen');
  const btnHeroStartCam = document.getElementById('btnHeroStartCam');
  const btnToggleCamera = document.getElementById('btnToggleCamera');
  const camPrompt = document.getElementById('camPrompt');
  const btnStartRealCam = document.getElementById('btnStartRealCam');
  const btnToggleHands = document.getElementById('btnToggleHands');
  const toggleModeBtn = document.getElementById('toggleModeBtn');
  const btnToggleTheater = document.getElementById('btnToggleTheater');
  const translationDualBox = document.querySelector('.translation-dual-box');
  const recentTagsBox = document.getElementById('recentTagsBox');
  const hudFpsVal = document.getElementById('hudFpsVal');
  const hudCamStatus = document.getElementById('hudCamStatus');
  const hudHandsStatus = document.getElementById('hudHandsStatus');
  const hudVoiceStatus = document.getElementById('hudVoiceStatus');
  const signerNameInput = document.getElementById('signerNameInput');
  const signerSavedBadge = document.getElementById('signerSavedBadge');
  const recentGesturesHistory = [];

  // Khởi tạo & đồng bộ tên người thử nghiệm (Signer Name)
  let signerDebounceTimer = null;
  if (signerNameInput) {
    signerNameInput.value = state.signerName;
    if (state.signerName) {
      fetch(`${API_BASE}/api/signer?name=${encodeURIComponent(state.signerName)}`, { method: 'POST' }).catch(() => {});
    }
    signerNameInput.addEventListener('input', (e) => {
      const val = e.target.value.trim();
      state.signerName = val;
      localStorage.setItem('vslr_signer_name', val);
      if (signerDebounceTimer) clearTimeout(signerDebounceTimer);
      signerDebounceTimer = setTimeout(() => {
        fetch(`${API_BASE}/api/signer?name=${encodeURIComponent(val || 'Khách')}`, { method: 'POST' }).catch(() => {});
        if (signerSavedBadge) {
          signerSavedBadge.classList.add('visible');
          setTimeout(() => signerSavedBadge.classList.remove('visible'), 1600);
        }
      }, 350);
    });
  }

  // =========================================================================
  // 3. NAVIGATION
  // =========================================================================
  const navItems = document.querySelectorAll('.nav-item[data-view]');
  const pageViews = document.querySelectorAll('.page-view');

  function navigateToView(viewId) {
    pageViews.forEach(view => {
      view.classList.toggle('active', view.id === viewId);
    });

    navItems.forEach(item => {
      item.classList.toggle('active', item.dataset.view === viewId);
    });

    state.activeView = viewId;
    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (viewId === 'view-translate') {
      updateCameraStateUI();
    }
  }

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navigateToView(item.dataset.view);
    });
  });

  document.getElementById('navLogo')?.addEventListener('click', () => {
    navigateToView('view-home');
  });

  document.getElementById('startTranslateBtn')?.addEventListener('click', () => {
    navigateToView('view-translate');
  });

  document.getElementById('exploreDictionaryBtn')?.addEventListener('click', () => {
    navigateToView('view-dictionary');
  });

  // =========================================================================
  // 4. TTS INTEGRATION (VieNeu-TTS - Giọng đọc Trúc Ly 48kHz)
  // =========================================================================
  function speakText(text) {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();

    const fallbackBrowserSpeech = () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = 'vi-VN';
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
      }
    };

    // Gọi API VieNeu-TTS backend với play_server=false để nhận luồng WAV phát trên loa trình duyệt
    fetch(`${API_BASE}/api/tts?text=${encodeURIComponent(cleanText)}&voice=${encodeURIComponent(state.ttsVoice)}&play_server=false`)
      .then(res => {
        const contentType = res.headers.get("content-type") || "";
        if (res.ok && contentType.includes("audio/wav")) {
          return res.blob().then(blob => {
            if (blob && blob.size > 100) {
              const audioUrl = URL.createObjectURL(blob);
              const audio = new Audio(audioUrl);
              audio.play().catch(() => fallbackBrowserSpeech());
              return;
            }
            fallbackBrowserSpeech();
          });
        }
        // Nếu server không trả về WAV (ví dụ đang tải model), phát ngay giọng đọc dự phòng trên trình duyệt
        fallbackBrowserSpeech();
      })
      .catch(() => {
        fallbackBrowserSpeech();
      });
  }

  // =========================================================================
  // 5. BACKEND REST API ACTIONS (SPACE, CLEAR, SPEAK, TOGGLE)
  // =========================================================================
  async function triggerBackendAction(actionName) {
    try {
      const res = await fetch(`${API_BASE}/api/action?action=${encodeURIComponent(actionName)}`, {
        method: 'POST'
      });
      return await res.json();
    } catch (err) {
      console.warn('Không thể gửi lệnh tới backend:', actionName, err);
    }
  }

  // YouTube Theater Mode (Phóng to video như YouTube)
  function toggleTheaterMode() {
    if (!translationDualBox) return;
    const isTheater = translationDualBox.classList.toggle('theater-mode');
    if (btnToggleTheater) {
      btnToggleTheater.classList.toggle('active', isTheater);
      btnToggleTheater.innerHTML = isTheater
        ? '<span class="theater-icon">🗗</span> Thu nhỏ (T)'
        : '<span class="theater-icon">⛶</span> Rạp chiếu (T)';
    }
  }

  btnToggleTheater?.addEventListener('click', toggleTheaterMode);

  // Khởi tạo luồng Webcam Trình duyệt (Client Webcam cho máy bạn bè từ xa)
  let frameCanvas = null;
  let clientWs = null;
  const handLandmarkCanvas = document.getElementById('handLandmarkCanvas');

  function drawHandSkeleton(ctx, pts, w, h, lineColor, pointColor) {
    const HAND_LINES = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17]
    ];
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2.5;
    HAND_LINES.forEach(([i, j]) => {
      if (pts[i] && pts[j]) {
        ctx.beginPath();
        ctx.moveTo(pts[i][0] * w, pts[i][1] * h);
        ctx.lineTo(pts[j][0] * w, pts[j][1] * h);
        ctx.stroke();
      }
    });
    ctx.fillStyle = pointColor;
    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * w, y * h, 4.0, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawSkeletonOverlay(skeleton) {
    if (!handLandmarkCanvas) return;
    if (!state.cameraEnabled || !state.showHands || !skeleton) {
      handLandmarkCanvas.classList.remove('active');
      handLandmarkCanvas.style.display = 'none';
      return;
    }

    if (liveVideo && liveVideo.videoWidth > 0) {
      if (handLandmarkCanvas.width !== liveVideo.videoWidth || handLandmarkCanvas.height !== liveVideo.videoHeight) {
        handLandmarkCanvas.width = liveVideo.videoWidth;
        handLandmarkCanvas.height = liveVideo.videoHeight;
      }
    }
    handLandmarkCanvas.classList.add('active');
    handLandmarkCanvas.style.display = 'block';
    const ctx = handLandmarkCanvas.getContext('2d');
    const w = handLandmarkCanvas.width || 640;
    const h = handLandmarkCanvas.height || 480;
    ctx.clearRect(0, 0, w, h);

    // 1. Vẽ khung xương thân người (Pose)
    if (skeleton.pose) {
      const p = skeleton.pose;
      ctx.strokeStyle = '#ffe119';
      ctx.lineWidth = 2.5;
      const poseLines = [
        ['11', '12'], ['11', '13'], ['13', '15'],
        ['12', '14'], ['14', '16'], ['11', '23'],
        ['12', '24'], ['23', '24']
      ];
      poseLines.forEach(([i, j]) => {
        if (p[i] && p[j]) {
          ctx.beginPath();
          ctx.moveTo(p[i][0] * w, p[i][1] * h);
          ctx.lineTo(p[j][0] * w, p[j][1] * h);
          ctx.stroke();
        }
      });
      ctx.fillStyle = '#3cb44b';
      Object.values(p).forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * w, y * h, 4.0, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // 2. Vẽ khung xương Bàn tay Trái (Đỏ / Cam)
    if (skeleton.left && skeleton.left.length === 21) {
      drawHandSkeleton(ctx, skeleton.left, w, h, '#f58231', '#e6194b');
    }

    // 3. Vẽ khung xương Bàn tay Phải (Xanh dương / Cyan)
    if (skeleton.right && skeleton.right.length === 21) {
      drawHandSkeleton(ctx, skeleton.right, w, h, '#46f0f0', '#0082c8');
    }
  }

  async function startClientWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false
      });
      state.clientStream = stream;
      if (liveVideo) {
        liveVideo.srcObject = stream;
        liveVideo.classList.remove('hidden');
        liveVideo.style.display = 'block';
        liveVideo.style.transform = 'scaleX(-1)';
        await liveVideo.play().catch(() => {});
      }
      if (liveStreamImg) {
        liveStreamImg.classList.add('hidden');
        liveStreamImg.style.display = 'none';
        liveStreamImg.removeAttribute('src');
      }
      state.cameraEnabled = true;
      updateCameraStateUI();
      startFrameStreamingLoop();
      triggerBackendAction('start_camera&client_mode=true');
    } catch (err) {
      console.error("Không thể mở Webcam trình duyệt:", err);
      alert("Vui lòng cấp quyền truy cập Webcam trên trình duyệt để nhận diện!");
    }
  }

  function stopClientWebcam() {
    if (state.clientFrameTimer) {
      clearInterval(state.clientFrameTimer);
      state.clientFrameTimer = null;
    }
    if (clientWs) {
      try { clientWs.close(); } catch (e) {}
      clientWs = null;
    }
    if (state.clientStream) {
      state.clientStream.getTracks().forEach(t => t.stop());
      state.clientStream = null;
    }
    if (liveVideo) {
      liveVideo.pause();
      liveVideo.srcObject = null;
      liveVideo.style.display = 'none';
    }
    if (handLandmarkCanvas) {
      handLandmarkCanvas.classList.remove('active');
      handLandmarkCanvas.style.display = 'none';
    }
    state.cameraEnabled = false;
    updateCameraStateUI();
    triggerBackendAction('stop_camera');
  }

  function toggleCameraAction() {
    if (state.clientCamMode || !isLocalPC) {
      if (state.cameraEnabled) {
        stopClientWebcam();
      } else {
        startClientWebcam();
      }
    } else {
      triggerBackendAction('toggle_camera');
    }
  }

  let lastSeenDecisionSeq = 0;

  function handleClientTelemetry(res) {
    if (!res) return;
    if (hudFpsVal && res.fps) {
      hudFpsVal.textContent = res.fps.toFixed(1);
    }
    if (hudHandsStatus && typeof res.hands !== 'undefined') {
      hudHandsStatus.textContent = `${res.hands} tay`;
      hudHandsStatus.style.color = res.hands > 0 ? '#38bdf8' : '#94a3b8';
    }
    if (hudCamStatus && res.camera) {
      hudCamStatus.textContent = res.camera;
    }

    // Cập nhật trạng thái GHI CỬ CHỈ (màu đỏ nhấp nháy) vs SẴN SÀNG
    if (res.in_segment) {
      recStatusTag?.classList.add('active');
      if (recStatusText) recStatusText.textContent = 'ĐANG GHI CỬ CHỈ...';
      if (btnMainToggleTranslate) {
        btnMainToggleTranslate.textContent = '[# DỪNG & DỊCH (SPACE)]';
        btnMainToggleTranslate.classList.add('stopping');
      }
    } else {
      recStatusTag?.classList.remove('active');
      if (recStatusText) recStatusText.textContent = 'SẴN SÀNG (SPACE)';
      if (btnMainToggleTranslate) {
        btnMainToggleTranslate.textContent = '[> BẮT ĐẦU GHI (SPACE)]';
        btnMainToggleTranslate.classList.remove('stopping');
      }
    }

    // Vẽ khung xương trực tiếp (0ms delay)
    if (res.skeleton !== undefined) {
      drawSkeletonOverlay(res.skeleton);
    }

    // Cập nhật kết quả dự đoán và phát âm thanh ngay lập tức
    if (res.decision && res.decision.seq && res.decision.seq > lastSeenDecisionSeq) {
      lastSeenDecisionSeq = res.decision.seq;
      handleBackendEvent(res.decision);
    }
  }

  function startFrameStreamingLoop() {
    if (state.clientFrameTimer) clearInterval(state.clientFrameTimer);
    const isFastNetwork = isDirectTunnel || API_BASE.includes('.trycloudflare.com') || isLocalPC;
    if (!frameCanvas) {
      frameCanvas = document.createElement('canvas');
    }
    frameCanvas.width = isFastNetwork ? 480 : 320;
    frameCanvas.height = isFastNetwork ? 360 : 240;
    const ctx = frameCanvas.getContext('2d');
    let isSending = false;
    let wsSafetyTimeout = null;

    // 1. Mở kết nối WebSocket tốc độ cao tới Backend
    try {
      if (clientWs) {
        try { clientWs.close(); } catch (e) {}
      }
      const baseOrigin = API_BASE || window.location.origin;
      const wsUrl = baseOrigin.replace(/^http/, 'ws') + '/api/ws/client_feed';
      clientWs = new WebSocket(wsUrl);
      clientWs.binaryType = 'arraybuffer';
      clientWs.onmessage = (evt) => {
        isSending = false;
        clearTimeout(wsSafetyTimeout);
        try {
          const res = JSON.parse(evt.data);
          handleClientTelemetry(res);
        } catch (err) {}
      };
    } catch (e) {
      clientWs = null;
    }

    // 2. Vòng lặp truyền frame Lock-Step (chống ứ đọng buffer: chỉ chụp và gửi khi server đã nhận xong frame trước)
    const streamInterval = isFastNetwork ? 35 : 55;
    const jpegQuality = isFastNetwork ? 0.65 : 0.50;

    const pumpNextFrame = () => {
      if (!state.cameraEnabled || !liveVideo || liveVideo.paused || liveVideo.ended || isSending) return;
      isSending = true;
      try {
        ctx.drawImage(liveVideo, 0, 0, frameCanvas.width, frameCanvas.height);
        frameCanvas.toBlob((blob) => {
          if (!blob) { isSending = false; return; }

          // Ưu tiên 1: Gửi qua WebSocket (0ms latency, không buffer)
          if (clientWs && clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(blob);
            clearTimeout(wsSafetyTimeout);
            wsSafetyTimeout = setTimeout(() => { isSending = false; }, 200);
            return;
          }

          // Ưu tiên 2: HTTP Fallback
          fetch(`${API_BASE}/api/client_frame?signer=${encodeURIComponent(state.signerName || 'Khách')}`, {
            method: 'POST',
            body: blob,
            headers: { 'Content-Type': 'image/jpeg' }
          })
          .then(r => r.json())
          .then(res => { handleClientTelemetry(res); })
          .catch(() => {})
          .finally(() => { isSending = false; });
        }, 'image/jpeg', jpegQuality);
      } catch (e) {
        isSending = false;
      }
    };

    state.clientFrameTimer = setInterval(pumpNextFrame, streamInterval);
  }

  // Phím bấm giao diện
  btnHeroStartCam?.addEventListener('click', toggleCameraAction);
  btnToggleCamera?.addEventListener('click', toggleCameraAction);

  btnMainToggleTranslate?.addEventListener('click', () => {
    if (!state.cameraEnabled) {
      toggleCameraAction();
    } else {
      triggerBackendAction('space');
    }
  });

  document.getElementById('btnClearSentence')?.addEventListener('click', () => triggerBackendAction('clear'));
  document.getElementById('btnSpeakSentence')?.addEventListener('click', () => triggerBackendAction('speak'));
  btnToggleHands?.addEventListener('click', () => triggerBackendAction('toggle_hands'));
  toggleModeBtn?.addEventListener('click', () => triggerBackendAction('toggle_mode'));

  // Phím tắt bàn phím (Tương thích 100% với CHAY_CAMERA_*.bat)
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Nếu đang mở popup chi tiết video hướng dẫn
    const isModalOpen = gestureDetailModal && gestureDetailModal.classList.contains('open');
    if (isModalOpen) {
      if (e.code === 'Space') {
        e.preventDefault();
        if (detailTutorialVideo) {
          if (detailTutorialVideo.paused) {
            detailTutorialVideo.play().catch(() => {});
          } else {
            detailTutorialVideo.pause();
          }
        }
        return;
      }
    }

    if (e.key === 'w' || e.key === 'W') {
      e.preventDefault();
      toggleCameraAction();
    } else if (e.code === 'Space') {
      e.preventDefault();
      if (state.activeView !== 'view-translate') {
        navigateToView('view-translate');
      }
      if (!state.cameraEnabled) {
        toggleCameraAction();
      } else {
        triggerBackendAction('space');
      }
    } else if (e.key === 't' || e.key === 'T') {
      e.preventDefault();
      toggleTheaterMode();
    } else if (e.key === 'm' || e.key === 'M') {
      e.preventDefault();
      triggerBackendAction('toggle_mode');
    } else if (e.key === 'h' || e.key === 'H') {
      e.preventDefault();
      triggerBackendAction('toggle_hands');
    } else if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      triggerBackendAction('speak');
    } else if (e.key === 'c' || e.key === 'C') {
      e.preventDefault();
      triggerBackendAction('clear');
    }
  });

  // =========================================================================
  // 6. REALTIME SSE TELEMETRY & PREDICTIONS STREAM
  // =========================================================================
  function initEventStream() {
    if (state.eventSource) {
      state.eventSource.close();
    }

    state.eventSource = new EventSource(`${API_BASE}/api/events`);

    state.eventSource.onopen = () => {
      state.backendConnected = true;
      camPrompt?.classList.add('hidden');
    };

    state.eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        handleBackendEvent(data);
      } catch (err) {
        // Non-json keepalive
      }
    };

    state.eventSource.onerror = () => {
      state.backendConnected = false;
      // Thử kết nối lại sau 2.5s
      setTimeout(() => {
        if (!state.backendConnected) initEventStream();
      }, 2500);
    };
  }

  function handleBackendEvent(data) {
    if (!data || !data.type) return;

    switch (data.type) {
      case 'init':
        state.cameraEnabled = !!data.camera_enabled;
        state.recMode = data.rec_mode || 'auto';
        state.showHands = !!data.show_hands;
        state.ttsVoice = data.tts_voice || 'Trúc Ly';
        state.sentence = data.sentence || [];
        if (data.signer && !state.signerName) {
          state.signerName = data.signer;
          if (signerNameInput) signerNameInput.value = data.signer;
        }
        updateCameraStateUI();
        updateModeUI();
        updateHandsUI();
        renderSentence();
        if (hudVoiceStatus) hudVoiceStatus.textContent = `${state.ttsVoice} (48kHz)`;
        if (hudCamStatus && data.camera && state.cameraEnabled) hudCamStatus.textContent = data.camera;
        break;

      case 'camera_state':
        state.cameraEnabled = !!data.enabled;
        if (data.client_mode) {
          state.clientCamMode = true;
        }
        updateCameraStateUI();
        if (state.cameraEnabled && data.name && hudCamStatus) {
          hudCamStatus.textContent = `[${data.index}] ${data.name}`;
        }
        break;

      case 'telemetry':
        if (typeof data.camera_enabled !== 'undefined' && state.cameraEnabled !== data.camera_enabled) {
          state.cameraEnabled = data.camera_enabled;
          updateCameraStateUI();
        }
        if (!state.cameraEnabled) break;

        if (hudFpsVal) hudFpsVal.textContent = data.fps ? data.fps.toFixed(1) : '30.0';
        if (hudCamStatus && data.camera) hudCamStatus.textContent = data.camera;
        if (hudHandsStatus) {
          hudHandsStatus.textContent = data.hands_count > 0 ? `${data.hands_count} tay` : '0 tay';
          hudHandsStatus.style.color = data.hands_count > 0 ? '#38bdf8' : '#94a3b8';
        }

        // Cập nhật trạng thái ghi nhận cử chỉ
        if (data.in_segment) {
          recStatusTag?.classList.add('active');
          if (recStatusText) {
            recStatusText.textContent = data.rec_mode === 'manual'
              ? `ĐANG GHI: ${data.rec_elapsed}s`
              : 'ĐANG GHI CỬ CHỈ...';
          }
          if (btnMainToggleTranslate) {
            btnMainToggleTranslate.textContent = '[# DỪNG & DỊCH (SPACE)]';
            btnMainToggleTranslate.classList.add('stopping');
          }
        } else {
          recStatusTag?.classList.remove('active');
          if (recStatusText) {
            recStatusText.textContent = data.rec_mode === 'manual'
              ? 'SẴN SÀNG (SPACE)'
              : 'TỰ ĐỘNG (Continuous)';
          }
          if (btnMainToggleTranslate) {
            btnMainToggleTranslate.textContent = '[> BẮT ĐẦU GHI (SPACE)]';
            btnMainToggleTranslate.classList.remove('stopping');
          }
        }
        break;

      case 'prediction_preview':
        if (heroPredictedWord) {
          heroPredictedWord.textContent = data.label;
          heroPredictedWord.style.opacity = '0.85';
        }
        if (predConfVal) {
          predConfVal.textContent = `${data.confidence}% (Đang nhận diện...)`;
        }
        if (recStatusText) {
          recStatusText.textContent = `ĐANG NHẬN DIỆN: ${data.label} (${data.confidence}%)`;
        }
        break;

      case 'prediction':
        if (data.accepted) {
          // Hiển thị chữ dự đoán lớn kèm hiệu ứng pop
          if (heroPredictedWord) {
            heroPredictedWord.style.opacity = '1.0';
            heroPredictedWord.textContent = data.label;
            heroPredictedWord.classList.add('pop');
            setTimeout(() => heroPredictedWord.classList.remove('pop'), 250);
          }
          if (predConfVal) predConfVal.textContent = `${data.confidence}%`;

          // Cập nhật câu
          state.sentence = data.sentence || [];
          renderSentence();

          // Cập nhật dải lịch sử nhận diện
          pushRecentGesture(data.label, data.confidence);

          if (recStatusText) {
            recStatusText.textContent = `ĐÃ DỊCH: ${data.label} (${data.confidence}%)`;
          }

          // Phát âm trên loa thiết bị người dùng (đặc biệt khi truy cập từ xa)
          speakText(data.label);
        } else {
          if (recStatusText) {
            recStatusText.textContent = `BỎ QUA: ${data.label} (${data.confidence}%) — ${data.reason}`;
          }
        }
        break;

      case 'sentence_updated':
        state.sentence = data.sentence || [];
        renderSentence();
        break;

      case 'sentence_spoken':
        if (recStatusText) {
          recStatusText.textContent = `ĐÃ PHÁT ÂM: "${data.text}" (${state.ttsVoice})`;
        }
        speakText(data.text);
        state.sentence = [];
        renderSentence();
        break;

      case 'settings':
        if (typeof data.show_hands !== 'undefined') {
          state.showHands = data.show_hands;
          updateHandsUI();
        }
        if (typeof data.rec_mode !== 'undefined') {
          state.recMode = data.rec_mode;
          updateModeUI();
        }
        break;

      case 'camera_switched':
        if (hudCamStatus && data.name) hudCamStatus.textContent = data.name;
        ensureLiveStreamRunning();
        break;
    }
  }

  function updateModeUI() {
    if (toggleModeBtn) {
      const isAuto = state.recMode === 'auto';
      toggleModeBtn.classList.toggle('active', isAuto);
      toggleModeBtn.textContent = isAuto ? 'Chế độ: Tự động (M)' : 'Chế độ: Thủ công (M)';
    }
  }

  function updateHandsUI() {
    if (btnToggleHands) {
      btnToggleHands.classList.toggle('active', state.showHands);
      btnToggleHands.textContent = state.showHands ? 'Khung xương: BẬT (H)' : 'Khung xương: TẮT (H)';
    }
  }

  function renderSentence() {
    if (!sentenceWordsBox) return;

    if (state.sentence.length === 0) {
      sentenceWordsBox.innerHTML = `<span class="empty-hint">Chưa có từ nào... (Thực hiện cử chỉ trước camera hoặc nhấn SPACE)</span>`;
      return;
    }

    sentenceWordsBox.innerHTML = state.sentence.map((w, idx) => `
      <span class="word-tag">
        ${w}
        <span class="word-tag-del" data-idx="${idx}">&times;</span>
      </span>
    `).join('');
  }

  function pushRecentGesture(label, conf) {
    if (!recentTagsBox) return;
    recentGesturesHistory.unshift({ label, conf, time: Date.now() });
    if (recentGesturesHistory.length > 8) {
      recentGesturesHistory.pop();
    }
    recentTagsBox.innerHTML = recentGesturesHistory.map(item => `
      <span class="recent-tag">
        <strong>${item.label}</strong>
        <span class="conf">${item.conf}%</span>
      </span>
    `).join('');
  }

  // Xóa từng từ trong câu
  sentenceWordsBox?.addEventListener('click', (e) => {
    if (e.target.classList.contains('word-tag-del')) {
      const idx = parseInt(e.target.dataset.idx, 10);
      state.sentence.splice(idx, 1);
      renderSentence();
      // Đồng bộ về backend
      fetch(`${API_BASE}/api/action?action=clear`, { method: 'POST' }).then(() => {
        state.sentence.forEach(w => {
          // preserve client edit
        });
      });
    }
  });

  // =========================================================================
  // 7. CAMERA STREAM & ON-DEMAND WEBCAM CONTROL
  // =========================================================================
  function updateCameraStateUI() {
    if (state.cameraEnabled) {
      camStandbyScreen?.classList.add('hidden');
      camPrompt?.classList.add('hidden');
      if (state.clientStream) {
        if (liveVideo) {
          liveVideo.classList.remove('hidden');
          liveVideo.style.display = 'block';
        }
        if (liveStreamImg) {
          liveStreamImg.classList.add('hidden');
          liveStreamImg.style.display = 'none';
        }
      } else {
        if (liveStreamImg) {
          liveStreamImg.classList.remove('hidden');
          liveStreamImg.style.display = 'block';
          const curSrc = liveStreamImg.getAttribute('src') || '';
          if (!curSrc.includes('/api/video_feed')) {
            liveStreamImg.src = `${API_BASE}/api/video_feed?t=` + Date.now();
          }
        }
      }
      if (recStatusTag) recStatusTag.style.display = 'flex';
      if (btnToggleCamera) {
        btnToggleCamera.classList.add('active');
        btnToggleCamera.textContent = '📷 Webcam: BẬT (W)';
      }
      if (btnMainToggleTranslate && !btnMainToggleTranslate.classList.contains('stopping')) {
        btnMainToggleTranslate.textContent = '[> BẮT ĐẦU GHI (SPACE)]';
      }
    } else {
      camStandbyScreen?.classList.remove('hidden');
      camPrompt?.classList.add('hidden');
      if (liveVideo) {
        liveVideo.classList.add('hidden');
        liveVideo.style.display = 'none';
      }
      if (liveStreamImg) {
        liveStreamImg.removeAttribute('src');
        liveStreamImg.classList.add('hidden');
        liveStreamImg.style.display = 'none';
      }
      if (recStatusTag) {
        recStatusTag.classList.remove('active');
        recStatusTag.style.display = 'none';
      }
      if (hudFpsVal) hudFpsVal.textContent = '0.0';
      if (hudCamStatus) hudCamStatus.textContent = state.clientStream ? 'Webcam Trình duyệt' : 'Chờ bật (Tắt)';
      if (hudHandsStatus) {
        hudHandsStatus.textContent = '0 tay';
        hudHandsStatus.style.color = '#94a3b8';
      }
      if (btnToggleCamera) {
        btnToggleCamera.classList.remove('active');
        btnToggleCamera.textContent = '📷 Webcam: TẮT (W)';
      }
      if (btnMainToggleTranslate) {
        btnMainToggleTranslate.classList.remove('stopping');
        btnMainToggleTranslate.textContent = '[▶ BẬT WEBCAM NHẬN DIỆN (W)]';
      }
    }
  }

  function ensureLiveStreamRunning() {
    if (!state.cameraEnabled || state.clientStream) return;
    if (liveStreamImg) {
      liveStreamImg.src = `${API_BASE}/api/video_feed?t=` + Date.now();
      camPrompt?.classList.add('hidden');
    }
  }

  btnStartRealCam?.addEventListener('click', toggleCameraAction);

  // Handle stream reload if image errors while camera is enabled
  liveStreamImg?.addEventListener('error', () => {
    if (!state.cameraEnabled || state.clientStream) return;
    camPrompt?.classList.remove('hidden');
    setTimeout(ensureLiveStreamRunning, 2000);
  });

  // =========================================================================
  // 8. DICTIONARY RENDERING & SEARCH (24 Gestures + Video Tutorials P01..P04)
  // =========================================================================
  const dictGridContainer = document.getElementById('dictGridContainer');
  const dictFilterInput = document.getElementById('dictFilterInput');
  const btnViewCards = document.getElementById('btnViewCards');
  const btnViewVideos = document.getElementById('btnViewVideos');

  function renderDictionary(filter = '') {
    if (!dictGridContainer) return;

    const q = filter.toLowerCase();
    const filtered = GESTURE_DATA.filter(g =>
      g.name.toLowerCase().includes(q) ||
      g.cat.toLowerCase().includes(q) ||
      g.signer.toLowerCase().includes(q)
    );

    if (state.dictViewMode === 'videos') {
      dictGridContainer.className = 'dict-grid mode-videos';
      dictGridContainer.innerHTML = filtered.map(g => `
        <div class="dict-video-card" data-gesture-id="${g.id}">
          <div class="dict-video-thumb-wrap">
            <img src="${g.thumb}" alt="${g.name}" loading="lazy" class="dict-video-thumb">
            <div class="dict-video-play-overlay">
              <span class="play-icon-circle">▶</span>
            </div>
            <span class="dict-video-duration">Signer ${g.signer}</span>
          </div>
          <div class="dict-video-info">
            <div class="dict-video-title">${g.icon} ${g.name}</div>
            <div class="dict-video-meta">
              <span class="dict-cat-tag">${g.cat} • ${g.signer}</span>
              <span class="dict-view-action">Xem hướng dẫn →</span>
            </div>
          </div>
        </div>
      `).join('');
    } else {
      dictGridContainer.className = 'dict-grid mode-cards';
      dictGridContainer.innerHTML = filtered.map(g => `
        <div class="dict-card" data-gesture-id="${g.id}">
          <div class="dict-card-left">
            <span class="dict-icon">${g.icon}</span>
            <div>
              <div class="dict-name">${g.name}</div>
              <div class="dict-cat">${g.cat}</div>
            </div>
          </div>
          <div class="dict-card-right">
            <span class="dict-video-pill">▶ ${g.signer}</span>
            <span class="dict-arrow">→</span>
          </div>
        </div>
      `).join('');
    }
  }

  renderDictionary();

  dictFilterInput?.addEventListener('input', (e) => {
    renderDictionary(e.target.value.trim());
  });

  btnViewCards?.addEventListener('click', () => {
    state.dictViewMode = 'cards';
    btnViewCards.classList.add('active');
    btnViewVideos?.classList.remove('active');
    renderDictionary(dictFilterInput ? dictFilterInput.value.trim() : '');
  });

  btnViewVideos?.addEventListener('click', () => {
    state.dictViewMode = 'videos';
    btnViewVideos.classList.add('active');
    btnViewCards?.classList.remove('active');
    renderDictionary(dictFilterInput ? dictFilterInput.value.trim() : '');
  });

  // =========================================================================
  // 9. MODALS (ABOUT & GESTURE DETAIL WITH VIDEO TUTORIAL)
  // =========================================================================
  const aboutModal = document.getElementById('aboutModal');
  const gestureDetailModal = document.getElementById('gestureDetailModal');
  const detailTutorialVideo = document.getElementById('detailTutorialVideo');
  const btnReplayTutorial = document.getElementById('btnReplayTutorial');
  const speedBtns = document.querySelectorAll('.speed-btn[data-speed]');

  document.getElementById('openAboutBtn')?.addEventListener('click', () => {
    aboutModal?.classList.add('open');
  });

  document.getElementById('closeAboutModal')?.addEventListener('click', () => {
    aboutModal?.classList.remove('open');
  });

  document.getElementById('closeAboutBtnBottom')?.addEventListener('click', () => {
    aboutModal?.classList.remove('open');
  });

  // Chi tiết cử chỉ & Video hướng dẫn
  let activeModalGesture = null;
  const detailCatBadge = document.getElementById('detailCatBadge');
  const detailSignerBadge = document.getElementById('detailSignerBadge');
  const detailGestureTitle = document.getElementById('detailGestureTitle');
  const detailGestureIcon = document.getElementById('detailGestureIcon');
  const detailGestureDesc = document.getElementById('detailGestureDesc');
  const detailGestureTip = document.getElementById('detailGestureTip');

  function openGestureDetail(g) {
    activeModalGesture = g;
    if (detailCatBadge) detailCatBadge.textContent = g.cat;
    if (detailSignerBadge) detailSignerBadge.textContent = `🎬 Video mẫu chuẩn: Signer ${g.signer}`;
    if (detailGestureTitle) detailGestureTitle.textContent = g.name;
    if (detailGestureIcon) detailGestureIcon.textContent = g.icon;
    if (detailGestureDesc) detailGestureDesc.textContent = g.desc;
    if (detailGestureTip) detailGestureTip.textContent = g.tip;

    if (detailTutorialVideo) {
      detailTutorialVideo.poster = g.thumb;
      detailTutorialVideo.src = g.video;
      detailTutorialVideo.currentTime = 0;
      detailTutorialVideo.playbackRate = state.currentSpeed || 1.0;
      detailTutorialVideo.play().catch(() => {});
    }

    gestureDetailModal?.classList.add('open');
  }

  function closeGestureDetail() {
    gestureDetailModal?.classList.remove('open');
    activeModalGesture = null;
    if (detailTutorialVideo) {
      detailTutorialVideo.pause();
      detailTutorialVideo.removeAttribute('src');
      detailTutorialVideo.load();
    }
  }

  document.getElementById('closeDetailModal')?.addEventListener('click', closeGestureDetail);

  document.addEventListener('click', (e) => {
    const card = e.target.closest('.dict-card, .dict-video-card');
    if (card) {
      const gid = card.dataset.gestureId;
      const found = GESTURE_DATA.find(g => g.id === gid);
      if (found) openGestureDetail(found);
    }
  });

  // Điều chỉnh tốc độ phát video hướng dẫn (0.5x, 0.75x, 1.0x)
  speedBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const spd = parseFloat(btn.dataset.speed) || 1.0;
      state.currentSpeed = spd;
      speedBtns.forEach(b => b.classList.toggle('active', b === btn));
      if (detailTutorialVideo) {
        detailTutorialVideo.playbackRate = spd;
      }
    });
  });

  // Xem lại từ đầu
  btnReplayTutorial?.addEventListener('click', () => {
    if (detailTutorialVideo) {
      detailTutorialVideo.currentTime = 0;
      detailTutorialVideo.playbackRate = state.currentSpeed || 1.0;
      detailTutorialVideo.play().catch(() => {});
    }
  });

  // Phát âm chuẩn VieNeu-TTS Trúc Ly từ trong Modal Từ điển
  document.getElementById('btnDetailSpeak')?.addEventListener('click', () => {
    if (activeModalGesture) speakText(activeModalGesture.name);
  });

  document.getElementById('btnDetailPractice')?.addEventListener('click', () => {
    closeGestureDetail();
    navigateToView('view-translate');
    triggerBackendAction('start_camera');
  });

  // Đóng modal khi click ra nền ngoài hoặc nhấn ESC
  aboutModal?.addEventListener('click', (e) => {
    if (e.target === aboutModal) aboutModal.classList.remove('open');
  });

  gestureDetailModal?.addEventListener('click', (e) => {
    if (e.target === gestureDetailModal) closeGestureDetail();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      aboutModal?.classList.remove('open');
      closeGestureDetail();
    }
  });

  // =========================================================================
  // 10. INITIALIZATION
  // =========================================================================
  updateCameraStateUI();
  initEventStream();

});
