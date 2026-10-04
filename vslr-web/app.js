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
    const cdnBase = (
      window.location.hostname === 'localhost' ||
      window.location.hostname === '127.0.0.1' ||
      window.location.hostname.endsWith('.vercel.app')
    ) ? '' : 'https://vslr-project-v3.vercel.app/';
    return {
      ...item,
      signer,
      video: `${cdnBase}tutorials/${item.id}.mp4?v=3.2`,
      thumb: `${cdnBase}tutorials/${item.id}.jpg?v=3.2`
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
  let API_BASE = isLocalHost ? '' : (localStorage.getItem('vslr_backend_url') || DEFAULT_REMOTE_BACKEND);

  const AUTO_DISCOVERY_TOPIC = 'vslr_ctu_aws_active_backend_prod_v3';
  const AUTO_DISCOVERY_POLL_URL = `https://ntfy.sh/${AUTO_DISCOVERY_TOPIC}/json?poll=1&since=24h`;
  const AUTO_DISCOVERY_SSE_URL = `https://ntfy.sh/${AUTO_DISCOVERY_TOPIC}/sse`;

  // Sinh mã phiên ngẫu nhiên độc lập cho từng tab / thiết bị (Multi-tenant)
  const clientSessionId = sessionStorage.getItem('vslr_session_id') || ('sess_' + Math.random().toString(36).substring(2, 10));
  sessionStorage.setItem('vslr_session_id', clientSessionId);

  const state = {
    activeView: 'view-home',
    dictViewMode: 'cards', // 'cards' | 'videos'
    currentSpeed: 1.0,
    cameraEnabled: false,
    clientCamMode: !isLocalPC,
    clientStream: null,
    clientFrameTimer: null,
    sessionId: clientSessionId,
    signerName: localStorage.getItem('vslr_signer_name') || '',
    isRecognizing: false, // Mặc định mở camera CHƯA nhận diện liền, chờ bấm nút bắt đầu
    targetGestureIndex: 0,
    targetGestureId: 'dich_tu_do',
    showTutorialPanel: false,
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
  const btnToggleRecognize = document.getElementById('btnToggleRecognize');
  const camPrompt = document.getElementById('camPrompt');
  const btnStartRealCam = document.getElementById('btnStartRealCam');
  const btnToggleTheater = document.getElementById('btnToggleTheater');
  const translationDualBox = document.querySelector('.translation-dual-box');
  const recentTagsBox = document.getElementById('recentTagsBox');
  const hudFpsVal = document.getElementById('hudFpsVal');
  const hudCamStatus = document.getElementById('hudCamStatus');
  const hudHandsStatus = document.getElementById('hudHandsStatus');
  const hudVoiceStatus = document.getElementById('hudVoiceStatus');
  const signerNameInput = document.getElementById('signerNameInput');
  const signerSavedBadge = document.getElementById('signerSavedBadge');
  const practiceGestureSelect = document.getElementById('practiceGestureSelect');
  const btnPrevGesture = document.getElementById('btnPrevGesture');
  const btnNextGesture = document.getElementById('btnNextGesture');
  const btnToggleTutorialPanel = document.getElementById('btnToggleTutorialPanel');
  const studyGestureIcon = document.getElementById('studyGestureIcon');
  const studyGestureTitle = document.getElementById('studyGestureTitle');
  const studyCatBadge = document.getElementById('studyCatBadge');
  const studyTutorialVideo = document.getElementById('studyTutorialVideo');
  const studyGestureDesc = document.getElementById('studyGestureDesc');
  const studyGestureTip = document.getElementById('studyGestureTip');
  const btnStudySpeakSample = document.getElementById('btnStudySpeakSample');
  const practiceMatchBadge = document.getElementById('practiceMatchBadge');
  const currentTargetFolderText = document.getElementById('currentTargetFolderText');
  const tutorialGuideCard = document.getElementById('tutorialGuideCard');
  const practiceStudioBox = document.getElementById('practiceStudioBox');
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
  // 2B. PRACTICE & LEARNING STUDIO (Video mẫu hướng dẫn nằm cạnh Webcam)
  // =========================================================================
  function setPracticeGesture(idx) {
    const cleanIdx = ((idx % GESTURE_DATA.length) + GESTURE_DATA.length) % GESTURE_DATA.length;
    state.targetGestureIndex = cleanIdx;
    const g = GESTURE_DATA[cleanIdx];
    state.targetGestureId = g.id;

    if (practiceGestureSelect) practiceGestureSelect.value = g.id;
    if (studyGestureIcon) studyGestureIcon.textContent = g.icon;
    if (studyGestureTitle) studyGestureTitle.textContent = g.name;
    if (studyCatBadge) studyCatBadge.textContent = `${g.cat} • Signer ${g.signer}`;
    if (studyGestureDesc) studyGestureDesc.textContent = g.desc;
    if (studyGestureTip) studyGestureTip.textContent = g.tip;
    if (currentTargetFolderText) currentTargetFolderText.textContent = `videos/${g.id}/`;

    if (practiceMatchBadge) {
      practiceMatchBadge.className = 'badge bg-info-subtle text-info-emphasis rounded-pill px-3 py-2';
      practiceMatchBadge.textContent = `Đang tập: ${g.name}`;
    }

    if (studyTutorialVideo) {
      studyTutorialVideo.poster = g.thumb;
      studyTutorialVideo.src = g.video;
      studyTutorialVideo.currentTime = 0;
      studyTutorialVideo.playbackRate = state.currentSpeed || 1.0;
      studyTutorialVideo.play().catch(() => {});
    }

    // Đồng bộ cử chỉ đang luyện tập về Backend để lưu đúng thư mục videos/<gesture_id>/
    fetch(`${API_BASE}/api/action?action=set_target_gesture&target_gesture=${encodeURIComponent(g.id)}&session_id=${encodeURIComponent(state.sessionId)}`, {
      method: 'POST'
    }).catch(() => {});
    if (clientWs && clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ target_gesture: g.id }));
    }
  }

  if (practiceGestureSelect) {
    practiceGestureSelect.innerHTML = GESTURE_DATA.map((g, i) => `
      <option value="${g.id}">${i + 1}/24: ${g.name} (${g.cat})</option>
    `).join('');
    practiceGestureSelect.addEventListener('change', (e) => {
      const idx = GESTURE_DATA.findIndex(item => item.id === e.target.value);
      if (idx >= 0) setPracticeGesture(idx);
    });
  }

  btnPrevGesture?.addEventListener('click', () => {
    setPracticeGesture(state.targetGestureIndex - 1);
  });

  btnNextGesture?.addEventListener('click', () => {
    setPracticeGesture(state.targetGestureIndex + 1);
  });

  const practiceSelectWrap = document.getElementById('practiceSelectWrap');
  const practiceNavBtns = document.getElementById('practiceNavBtns');

  btnToggleTutorialPanel?.addEventListener('click', () => {
    state.showTutorialPanel = !state.showTutorialPanel;
    if (tutorialGuideCard) {
      tutorialGuideCard.style.display = state.showTutorialPanel ? 'flex' : 'none';
    }
    if (practiceSelectWrap) {
      practiceSelectWrap.style.setProperty('display', state.showTutorialPanel ? 'flex' : 'none', 'important');
    }
    if (practiceNavBtns) {
      practiceNavBtns.style.setProperty('display', state.showTutorialPanel ? 'flex' : 'none', 'important');
    }
    if (practiceMatchBadge) {
      practiceMatchBadge.style.display = state.showTutorialPanel ? 'inline-block' : 'none';
    }
    if (practiceStudioBox) {
      practiceStudioBox.classList.toggle('single-column-studio', !state.showTutorialPanel);
    }
    if (btnToggleTutorialPanel) {
      btnToggleTutorialPanel.classList.toggle('active', state.showTutorialPanel);
      btnToggleTutorialPanel.textContent = state.showTutorialPanel ? '🎬 Đóng Video Mẫu' : '🎓 Mở Video Mẫu Song Song';
    }

    // Khi ẩn luyện tập: chuyển sang thư mục videos/dich_tu_do/, không lưu vào xin_chao nữa
    const targetFolder = state.showTutorialPanel ? state.targetGestureId : 'dich_tu_do';
    if (currentTargetFolderText) {
      currentTargetFolderText.textContent = state.recordConsent ? `videos/${targetFolder}/` : 'Đã tắt lưu video';
    }
    fetch(`${API_BASE}/api/action?action=set_target_gesture&target_gesture=${encodeURIComponent(targetFolder)}&session_id=${encodeURIComponent(state.sessionId)}`, {
      method: 'POST'
    }).catch(() => {});
    if (clientWs && clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ target_gesture: targetFolder }));
    }
  });

  // =========================================================================
  // 2C. RECORD CONSENT & USER CLIP REPLAY (XEM LẠI VIDEO VỪA THỰC HIỆN)
  // =========================================================================
  const consentModal = document.getElementById('consentModal');
  const btnConsentAccept = document.getElementById('btnConsentAccept');
  const btnConsentDecline = document.getElementById('btnConsentDecline');
  const btnToggleRecordConsent = document.getElementById('btnToggleRecordConsent');
  const btnReplayUserClip = document.getElementById('btnReplayUserClip');
  const userReplayModal = document.getElementById('userReplayModal');
  const userReplayVideo = document.getElementById('userReplayVideo');
  const closeUserReplayModal = document.getElementById('closeUserReplayModal');
  const btnCloseReplayBottom = document.getElementById('btnCloseReplayBottom');

  state.recordConsent = localStorage.getItem('vslr_record_consent') !== 'false';

  function updateRecordConsentUI() {
    if (btnToggleRecordConsent) {
      btnToggleRecordConsent.classList.toggle('active', state.recordConsent);
      btnToggleRecordConsent.textContent = state.recordConsent ? '💾 Lưu video: BẬT' : '💾 Lưu video: TẮT';
    }
    const targetFolder = state.showTutorialPanel ? state.targetGestureId : 'dich_tu_do';
    if (currentTargetFolderText) {
      currentTargetFolderText.textContent = state.recordConsent ? `videos/${targetFolder}/` : 'Không lưu (Riêng tư)';
    }
    fetch(`${API_BASE}/api/action?action=set_record_consent&consent=${state.recordConsent ? 'true' : 'false'}&session_id=${encodeURIComponent(state.sessionId)}`, {
      method: 'POST'
    }).catch(() => {});
  }

  // Hiển thị hộp thoại xin phép lưu video lần đầu tiên người dùng mở web
  if (localStorage.getItem('vslr_record_consent') === null && consentModal) {
    consentModal.classList.add('open');
  }
  setTimeout(updateRecordConsentUI, 100);

  btnConsentAccept?.addEventListener('click', () => {
    state.recordConsent = true;
    localStorage.setItem('vslr_record_consent', 'true');
    consentModal?.classList.remove('open');
    updateRecordConsentUI();
  });

  btnConsentDecline?.addEventListener('click', () => {
    state.recordConsent = false;
    localStorage.setItem('vslr_record_consent', 'false');
    consentModal?.classList.remove('open');
    updateRecordConsentUI();
  });

  btnToggleRecordConsent?.addEventListener('click', () => {
    state.recordConsent = !state.recordConsent;
    localStorage.setItem('vslr_record_consent', state.recordConsent ? 'true' : 'false');
    updateRecordConsentUI();
  });

  // Ghi nhận đoạn clip cử chỉ phía trình duyệt để người dùng bấm "Xem lại vừa làm" tức thì (0ms)
  let userMediaRecorder = null;
  let userRecordedChunks = [];
  let lastUserClipUrl = null;

  function startUserClipRecorder() {
    if (!state.clientStream || !window.MediaRecorder) return;
    try {
      if (userMediaRecorder && userMediaRecorder.state !== 'inactive') {
        userMediaRecorder.stop();
      }
      userRecordedChunks = [];
      userMediaRecorder = new MediaRecorder(state.clientStream);
      userMediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) userRecordedChunks.push(e.data);
      };
      userMediaRecorder.onstop = () => {
        if (userRecordedChunks.length > 0) {
          const blob = new Blob(userRecordedChunks, { type: userMediaRecorder.mimeType || 'video/webm' });
          if (lastUserClipUrl) URL.revokeObjectURL(lastUserClipUrl);
          lastUserClipUrl = URL.createObjectURL(blob);
          if (btnReplayUserClip) btnReplayUserClip.style.display = 'inline-block';
        }
      };
      userMediaRecorder.start(100);
    } catch (e) {
      userMediaRecorder = null;
    }
  }

  function finishUserClipRecorder() {
    if (userMediaRecorder && userMediaRecorder.state === 'recording') {
      try { userMediaRecorder.stop(); } catch (e) {}
    }
    if (state.isRecognizing) {
      setTimeout(startUserClipRecorder, 250);
    }
  }

  btnReplayUserClip?.addEventListener('click', () => {
    if (lastUserClipUrl && userReplayVideo) {
      userReplayVideo.src = lastUserClipUrl;
      userReplayModal?.classList.add('open');
      userReplayVideo.currentTime = 0;
      userReplayVideo.play().catch(() => {});
    }
  });

  function closeUserReplay() {
    userReplayModal?.classList.remove('open');
    if (userReplayVideo) {
      userReplayVideo.pause();
      userReplayVideo.removeAttribute('src');
    }
  }

  closeUserReplayModal?.addEventListener('click', closeUserReplay);
  btnCloseReplayBottom?.addEventListener('click', closeUserReplay);
  userReplayModal?.addEventListener('click', (e) => {
    if (e.target === userReplayModal) closeUserReplay();
  });

  btnStudySpeakSample?.addEventListener('click', () => {
    const g = GESTURE_DATA[state.targetGestureIndex];
    if (g) {
      lastSpokenTime = 0;
      speakText(g.name);
    }
  });

  document.querySelectorAll('.study-speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const spd = parseFloat(btn.dataset.speed) || 1.0;
      state.currentSpeed = spd;
      document.querySelectorAll('.study-speed-btn').forEach(b => b.classList.toggle('active', b === btn));
      if (studyTutorialVideo) studyTutorialVideo.playbackRate = spd;
    });
  });

  // Khởi tạo bài học đầu tiên (Xin chào)
  setTimeout(() => setPracticeGesture(0), 50);

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
  // 4. TTS INTEGRATION (VieNeu-TTS - Giọng đọc Trúc Ly 48kHz Duy Nhất)
  // =========================================================================
  let activeAudio = null;
  let activeTtsAbort = null;
  let ttsRequestId = 0;
  let lastSpokenText = '';
  let lastSpokenTime = 0;

  function speakText(text) {
    if (!text || !text.trim()) return;
    // Chống phát đè khi tab đang ẩn ở nền
    if (document.hidden) return;

    const cleanText = text.trim();
    const nowMs = Date.now();
    if (cleanText === lastSpokenText && (nowMs - lastSpokenTime) < 1500) {
      return;
    }
    lastSpokenText = cleanText;
    lastSpokenTime = nowMs;

    // 1. Tăng ID phiên phát âm & Hủy bỏ lượt tải âm thanh cũ
    ttsRequestId += 1;
    const currentReqId = ttsRequestId;
    if (activeTtsAbort) {
      try { activeTtsAbort.abort(); } catch (e) {}
      activeTtsAbort = null;
    }

    // 2. Dừng ngay lập tức âm thanh cũ đang phát
    if (activeAudio) {
      try {
        activeAudio.pause();
        activeAudio.src = '';
      } catch (e) {}
      activeAudio = null;
    }
    if ('speechSynthesis' in window) {
      try { window.speechSynthesis.cancel(); } catch (e) {}
    }

    activeTtsAbort = new AbortController();

    // 3. Chỉ dùng DUY NHẤT giọng đọc VieNeu-TTS Trúc Ly chuẩn 48kHz
    fetch(`${API_BASE}/api/tts?text=${encodeURIComponent(cleanText)}&voice=${encodeURIComponent(state.ttsVoice)}&play_server=false`, {
      signal: activeTtsAbort.signal
    })
      .then(res => {
        if (currentReqId !== ttsRequestId) return null;
        const contentType = res.headers.get("content-type") || "";
        if (res.ok && contentType.includes("audio/wav")) {
          return res.blob();
        }
        return null;
      })
      .then(blob => {
        if (!blob || currentReqId !== ttsRequestId || blob.size < 100) return;
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        activeAudio = audio;
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          if (activeAudio === audio) activeAudio = null;
        };
        audio.play().catch(() => {});
      })
      .catch(() => {});
  }

  // =========================================================================
  // 5. BACKEND REST API ACTIONS (SPACE, CLEAR, SPEAK, TOGGLE)
  // =========================================================================
  async function triggerBackendAction(actionName) {
    try {
      const parts = String(actionName).split('&');
      const mainAction = parts[0];
      const extra = parts.slice(1).map(p => '&' + p).join('');
      const url = `${API_BASE}/api/action?action=${encodeURIComponent(mainAction)}&session_id=${encodeURIComponent(state.sessionId)}${extra}`;
      const res = await fetch(url, {
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
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, min: 20 },
          facingMode: "user"
        },
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
    state.isRecognizing = false;
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
    triggerBackendAction(`stop_camera&session_id=${encodeURIComponent(state.sessionId)}`);
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

  function updateRecognizeUI() {
    if (!state.cameraEnabled) {
      if (btnToggleRecognize) {
        btnToggleRecognize.classList.remove('active');
        btnToggleRecognize.style.display = 'none';
      }
      if (recStatusTag) recStatusTag.style.display = 'none';
      if (btnMainToggleTranslate) {
        btnMainToggleTranslate.classList.remove('stopping');
        btnMainToggleTranslate.textContent = '[▶ BẬT WEBCAM NHẬN DIỆN (W)]';
      }
    } else if (!state.isRecognizing) {
      if (btnToggleRecognize) {
        btnToggleRecognize.style.display = 'inline-block';
        btnToggleRecognize.classList.remove('active');
        btnToggleRecognize.textContent = '▶ BẮT ĐẦU NHẬN DIỆN (SPACE)';
      }
      if (recStatusTag) {
        recStatusTag.style.display = 'flex';
        recStatusTag.classList.remove('active');
      }
      if (recStatusText) recStatusText.textContent = '⚪ ĐÃ BẬT CAMERA (Chờ bấm Bắt đầu)';
      if (btnMainToggleTranslate) {
        btnMainToggleTranslate.classList.remove('stopping');
        btnMainToggleTranslate.textContent = '[▶ BẮT ĐẦU NHẬN DIỆN (SPACE)]';
      }
    } else {
      if (btnToggleRecognize) {
        btnToggleRecognize.style.display = 'inline-block';
        btnToggleRecognize.classList.add('active');
        btnToggleRecognize.textContent = '⏸ TẠM DỪNG NHẬN DIỆN (SPACE)';
      }
      if (recStatusTag) {
        recStatusTag.style.display = 'flex';
        recStatusTag.classList.add('active');
      }
      if (recStatusText) recStatusText.textContent = '🟢 ĐANG NHẬN DIỆN (Sẵn sàng dơ tay)';
      if (btnMainToggleTranslate) {
        btnMainToggleTranslate.classList.add('stopping');
        btnMainToggleTranslate.textContent = '[⏸ TẠM DỪNG NHẬN DIỆN (SPACE)]';
      }
    }
  }

  function toggleRecognitionAction() {
    if (!state.cameraEnabled) {
      toggleCameraAction();
      return;
    }
    state.isRecognizing = !state.isRecognizing;
    updateRecognizeUI();
    if (state.isRecognizing) {
      startUserClipRecorder();
    } else {
      finishUserClipRecorder();
    }
    const act = state.isRecognizing ? 'start_recognize' : 'stop_recognize';
    triggerBackendAction(`${act}&session_id=${encodeURIComponent(state.sessionId)}`);
    if (clientWs && clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ recognizing: state.isRecognizing, signer: state.signerName }));
    }
  }

  let lastSeenDecisionSeq = 0;
  const btnChangeBackend = document.getElementById('btnChangeBackend');

  function setBackendStatusUI(connected) {
    state.backendConnected = connected;
    if (!btnChangeBackend) return;
    if (connected) {
      btnChangeBackend.style.background = '#ecfdf5';
      btnChangeBackend.style.color = '#059669';
      btnChangeBackend.style.borderColor = '#a7f3d0';
      btnChangeBackend.textContent = '🟢 AI Server: Đã kết nối';
    } else {
      btnChangeBackend.style.background = '#fef2f2';
      btnChangeBackend.style.color = '#dc2626';
      btnChangeBackend.style.borderColor = '#fecaca';
      btnChangeBackend.textContent = '🔴 Mất kết nối AI (Bấm đổi link)';
    }
  }

  btnChangeBackend?.addEventListener('click', (e) => {
    e.stopPropagation();
    const current = localStorage.getItem('vslr_backend_url') || API_BASE || '';
    const newUrl = prompt(
      'Nhập đường link Cloudflare mới từ cửa sổ Terminal AWS của bạn:\n(Ví dụ: https://xxxx-yyyy-zzzz.trycloudflare.com)',
      current
    );
    if (newUrl !== null && newUrl.trim()) {
      const cleaned = newUrl.trim().replace(/\/+$/, '');
      localStorage.setItem('vslr_backend_url', cleaned);
      window.location.search = `?backend=${encodeURIComponent(cleaned)}`;
    }
  });

  function handleClientTelemetry(res) {
    if (!res) return;
    setBackendStatusUI(true);
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

    if (typeof res.is_recognizing !== 'undefined' && state.isRecognizing !== res.is_recognizing) {
      state.isRecognizing = res.is_recognizing;
      updateRecognizeUI();
    }

    // Cập nhật trạng thái GHI CỬ CHỈ (màu đỏ) vs SẴN SÀNG (màu xanh)
    if (state.isRecognizing) {
      if (res.in_segment) {
        recStatusTag?.classList.add('active');
        if (recStatusText) recStatusText.textContent = '🔴 ĐANG GHI CỬ CHỈ...';
      } else {
        recStatusTag?.classList.remove('active');
        if (recStatusText) recStatusText.textContent = '🟢 SẴN SÀNG NHẬN DIỆN (Hãy dơ tay lên)';
      }
    }

    // Tắt vẽ khung xương trên giao diện web theo yêu cầu (tiết kiệm CPU & gọn gàng)
    // if (res.skeleton !== undefined) { drawSkeletonOverlay(res.skeleton); }

    // Cập nhật câu thoại đã dịch ngay khi có kết quả
    if (res.sentence && Array.isArray(res.sentence) && res.sentence.length !== state.sentence.length) {
      state.sentence = res.sentence;
      renderSentence();
    }

    // Cập nhật kết quả dự đoán và phát âm thanh ngay lập tức
    if (res.decision && res.decision.seq && res.decision.seq > lastSeenDecisionSeq) {
      handleBackendEvent(res.decision);
    }
  }

  function startFrameStreamingLoop() {
    if (state.clientFrameTimer) clearInterval(state.clientFrameTimer);
    if (!frameCanvas) {
      frameCanvas = document.createElement('canvas');
    }
    frameCanvas.width = 320;
    frameCanvas.height = 240;
    const ctx = frameCanvas.getContext('2d');
    let inFlight = 0;
    const MAX_IN_FLIGHT = 3;
    let wsSafetyTimeout = null;

    // 1. Mở kết nối WebSocket tốc độ cao Pipelined tới Backend (kèm session_id độc lập)
    try {
      if (clientWs) {
        try { clientWs.close(); } catch (e) {}
      }
      const baseOrigin = API_BASE || window.location.origin;
      const targetParam = (state.showTutorialPanel && state.targetGestureId && state.targetGestureId !== 'dich_tu_do')
        ? `&target_gesture=${encodeURIComponent(state.targetGestureId)}`
        : '&target_gesture=dich_tu_do';
      const wsUrl = baseOrigin.replace(/^http/, 'ws') +
        `/api/ws/client_feed?signer=${encodeURIComponent(state.signerName || 'Khách')}` +
        `&recognizing=${state.isRecognizing ? 1 : 0}` +
        `&session_id=${encodeURIComponent(state.sessionId)}` +
        targetParam;
      clientWs = new WebSocket(wsUrl);
      clientWs.binaryType = 'arraybuffer';
      clientWs.onmessage = (evt) => {
        if (inFlight > 0) inFlight--;
        clearTimeout(wsSafetyTimeout);
        try {
          const res = JSON.parse(evt.data);
          handleClientTelemetry(res);
        } catch (err) {}
        // Gọi ngay frame tiếp theo khi vừa nhận xong kết quả (Pipelined 0ms delay)
        requestAnimationFrame(pumpNextFrame);
      };
      clientWs.onerror = () => {
        inFlight = 0;
        clearTimeout(wsSafetyTimeout);
      };
      clientWs.onclose = () => {
        inFlight = 0;
        clearTimeout(wsSafetyTimeout);
      };
    } catch (e) {
      clientWs = null;
    }

    // 2. Vòng lặp truyền frame Pipelined (Định dạng nén siêu nhẹ 320x240, ~6KB)
    const jpegQuality = 0.50;

    const pumpNextFrame = () => {
      if (!state.cameraEnabled || !liveVideo || liveVideo.paused || liveVideo.ended || !state.isRecognizing || inFlight >= MAX_IN_FLIGHT) return;

      inFlight++;
      try {
        const vw = liveVideo.videoWidth || 640;
        const vh = liveVideo.videoHeight || 480;
        const isPortrait = vh > vw;

        // Tự động khớp khung hình theo đúng chiều Điện thoại dọc (240x320) hoặc Máy tính ngang (320x240)
        // để giữ trọn vẹn 100% khung hình từ đầu xuống thắt lưng, không cắt mất vùng bụng/thắt lưng
        if (isPortrait) {
          if (frameCanvas.width !== 240 || frameCanvas.height !== 320) {
            frameCanvas.width = 240;
            frameCanvas.height = 320;
          }
        } else {
          if (frameCanvas.width !== 320 || frameCanvas.height !== 240) {
            frameCanvas.width = 320;
            frameCanvas.height = 240;
          }
        }

        ctx.drawImage(liveVideo, 0, 0, frameCanvas.width, frameCanvas.height);
        frameCanvas.toBlob((blob) => {
          if (!blob) {
            if (inFlight > 0) inFlight--;
            return;
          }

          // Ưu tiên 1: Gửi qua WebSocket (0ms latency, Pipelined)
          if (clientWs && clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(blob);
            clearTimeout(wsSafetyTimeout);
            wsSafetyTimeout = setTimeout(() => { inFlight = 0; }, 1200);
            return;
          }

          // Ưu tiên 2: HTTP Fallback (có timeout 2s chống treo cứng)
          const controller = new AbortController();
          const tid = setTimeout(() => controller.abort(), 2000);
          fetch(`${API_BASE}/api/client_frame?signer=${encodeURIComponent(state.signerName || 'Khách')}&recognizing=${state.isRecognizing ? 1 : 0}&session_id=${encodeURIComponent(state.sessionId)}`, {
            method: 'POST',
            body: blob,
            headers: { 'Content-Type': 'image/jpeg' },
            signal: controller.signal
          })
          .then(r => r.json())
          .then(res => { handleClientTelemetry(res); })
          .catch(() => { setBackendStatusUI(false); })
          .finally(() => {
            clearTimeout(tid);
            if (inFlight > 0) inFlight--;
          });
        }, 'image/jpeg', jpegQuality);
      } catch (e) {
        if (inFlight > 0) inFlight--;
      }
    };

    state.clientFrameTimer = setInterval(pumpNextFrame, 25);
  }

  // Phím bấm giao diện
  btnHeroStartCam?.addEventListener('click', toggleCameraAction);
  btnToggleCamera?.addEventListener('click', toggleCameraAction);
  btnToggleRecognize?.addEventListener('click', toggleRecognitionAction);

  btnMainToggleTranslate?.addEventListener('click', () => {
    if (!state.cameraEnabled) {
      toggleCameraAction();
    } else {
      toggleRecognitionAction();
    }
  });

  function handleManualSpeak() {
    const textToSpeak = state.sentence.length > 0
      ? state.sentence.join(' ')
      : (heroPredictedWord && heroPredictedWord.textContent !== '---' ? heroPredictedWord.textContent : 'Xin chào');
    lastSpokenTime = 0; // Cho phép phát ngay khi bấm nút chủ động
    speakText(textToSpeak);
  }

  document.getElementById('btnClearSentence')?.addEventListener('click', () => triggerBackendAction(`clear&session_id=${encodeURIComponent(state.sessionId)}`));
  document.getElementById('btnSpeakSentence')?.addEventListener('click', handleManualSpeak);

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
        toggleRecognitionAction();
      }
    } else if (e.key === 't' || e.key === 'T') {
      e.preventDefault();
      toggleTheaterMode();
    } else if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      handleManualSpeak();
    } else if (e.key === 'c' || e.key === 'C') {
      e.preventDefault();
      triggerBackendAction(`clear&session_id=${encodeURIComponent(state.sessionId)}`);
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
      setBackendStatusUI(true);
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
      setBackendStatusUI(false);
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

        // Cập nhật trạng thái ghi nhận cử chỉ nếu người dùng đã bật nhận diện
        if (state.isRecognizing) {
          if (data.in_segment) {
            recStatusTag?.classList.add('active');
            if (recStatusText) recStatusText.textContent = '🔴 ĐANG GHI CỬ CHỈ...';
          } else {
            recStatusTag?.classList.remove('active');
            if (recStatusText) recStatusText.textContent = '🟢 ĐANG NHẬN DIỆN (Sẵn sàng dơ tay)';
          }
        } else {
          recStatusTag?.classList.remove('active');
          if (recStatusText) recStatusText.textContent = '⚪ ĐÃ BẬT CAMERA (Chờ bấm Bắt đầu)';
        }
        break;

      case 'prediction':
        if (data.seq) {
          if (data.seq <= lastSeenDecisionSeq) break;
          lastSeenDecisionSeq = data.seq;
        }
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

          // Kiểm tra và hiển thị kết quả chúc mừng nếu làm đúng cử chỉ đang luyện tập
          const curG = GESTURE_DATA[state.targetGestureIndex];
          if (curG && data.label && data.label.toLowerCase().trim() === curG.name.toLowerCase().trim()) {
            if (practiceMatchBadge) {
              practiceMatchBadge.className = 'badge bg-success rounded-pill px-3 py-2 animate__pulse';
              practiceMatchBadge.textContent = `🎉 CHÍNH XÁC: ${data.label} (${data.confidence}%)`;
            }
          } else if (practiceMatchBadge && curG) {
            practiceMatchBadge.className = 'badge bg-warning text-dark rounded-pill px-3 py-2';
            practiceMatchBadge.textContent = `💡 Phát hiện: ${data.label} (${data.confidence}%) — Đang tập: ${curG.name}`;
          }

          // Phát âm trên loa thiết bị người dùng (đặc biệt khi truy cập từ xa)
          speakText(data.label);

          // Hoàn tất lưu video clip của người dùng để có thể bấm "Xem lại vừa làm"
          finishUserClipRecorder();
        } else {
          // Khi nhận diện bị từ chối (reject): Vẫn hiển thị tên cử chỉ và số % trên màn hình webcam (Image #6)
          if (heroPredictedWord) {
            heroPredictedWord.style.opacity = '1.0';
            heroPredictedWord.textContent = `${data.label} (Nghi ngờ)`;
            heroPredictedWord.classList.add('pop');
            setTimeout(() => heroPredictedWord.classList.remove('pop'), 250);
          }
          if (predConfVal) {
            predConfVal.textContent = `${data.confidence}% (Từ chối)`;
          }

          if (recStatusText) {
            recStatusText.textContent = `BỎ QUA: ${data.label} (${data.confidence}%) — ${data.reason}`;
          }

          const curG = GESTURE_DATA[state.targetGestureIndex];
          if (practiceMatchBadge && curG) {
            practiceMatchBadge.className = 'badge bg-warning text-dark rounded-pill px-3 py-2';
            practiceMatchBadge.textContent = `⚠️ Nghi ngờ: ${data.label} (${data.confidence}%) — ${data.reason || 'Chưa đủ ngưỡng'}`;
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
      if (btnToggleCamera) {
        btnToggleCamera.classList.add('active');
        btnToggleCamera.textContent = '📷 Webcam: BẬT (W)';
      }
      updateRecognizeUI();
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
      updateRecognizeUI();
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
  // 9. MODALS (GESTURE DETAIL WITH VIDEO TUTORIAL)
  // =========================================================================
  const gestureDetailModal = document.getElementById('gestureDetailModal');
  const detailTutorialVideo = document.getElementById('detailTutorialVideo');
  const btnReplayTutorial = document.getElementById('btnReplayTutorial');
  const speedBtns = document.querySelectorAll('.speed-btn[data-speed]');

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
    const targetId = activeModalGesture ? activeModalGesture.id : 'xin_chao';
    closeGestureDetail();
    const targetIdx = GESTURE_DATA.findIndex(g => g.id === targetId);
    if (targetIdx >= 0) setPracticeGesture(targetIdx);
    navigateToView('view-translate');
    if (!state.cameraEnabled) toggleCameraAction();
  });

  // Đóng modal khi click ra nền ngoài hoặc nhấn ESC
  gestureDetailModal?.addEventListener('click', (e) => {
    if (e.target === gestureDetailModal) closeGestureDetail();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeGestureDetail();
    }
  });

  // =========================================================================
  // 10. INITIALIZATION & REALTIME AUTO-DISCOVERY (TỰ ĐỘNG ĐỒNG BỘ LINK AWS)
  // =========================================================================
  function applyDiscoveredBackend(newUrl) {
    if (!newUrl || typeof newUrl !== 'string') return;
    const cleaned = newUrl.trim().replace(/\/+$/, '');
    if (!cleaned.startsWith('https://') && !cleaned.startsWith('http://')) return;
    const changed = (cleaned !== API_BASE);
    API_BASE = cleaned;
    localStorage.setItem('vslr_backend_url', cleaned);
    if (changed || !state.backendConnected) {
      initEventStream();
      if (state.cameraEnabled) {
        startFrameStreamingLoop();
      }
    }
  }

  async function startAutoDiscovery() {
    if (isLocalHost) {
      setBackendStatusUI(true);
      return;
    }

    // 1. Tự động lấy link Cloudflare mới nhất từ kho Auto-Discovery ngay khi vừa mở web
    try {
      const resp = await fetch(AUTO_DISCOVERY_POLL_URL, { cache: 'no-store' });
      if (resp.ok) {
        const text = await resp.text();
        const lines = text.trim().split('\n');
        let latestTime = -1;
        let latestUrl = '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (
              data.event === 'message' &&
              data.message &&
              data.message.startsWith('https://') &&
              data.message.includes('.trycloudflare.com') &&
              (data.time || 0) >= latestTime
            ) {
              latestTime = data.time || 0;
              latestUrl = data.message.trim();
            }
          } catch (e) {}
        }
        if (latestUrl) {
          applyDiscoveredBackend(latestUrl);
        }
      }
    } catch (err) {}

    // 2. Lắng nghe trực tiếp (Live SSE): Hễ bạn bật/khởi động lại AWS là web tự nhận link mới trong 0.2s (không cần F5!)
    try {
      const discoSource = new EventSource(AUTO_DISCOVERY_SSE_URL);
      discoSource.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (
            data.event === 'message' &&
            data.message &&
            data.message.startsWith('https://') &&
            data.message.includes('.trycloudflare.com')
          ) {
            applyDiscoveredBackend(data.message.trim());
          }
        } catch (err) {}
      };
    } catch (e) {}
  }

  // =========================================================================
  // 11. TRANSLATE MODE SWITCHER & VIDEO IMPORT RECOGNITION STUDIO
  // =========================================================================
  const tabModeLiveCam = document.getElementById('tabModeLiveCam');
  const tabModeImportVideo = document.getElementById('tabModeImportVideo');
  const videoImportStudio = document.getElementById('videoImportStudio');

  // Video Import Elements
  const videoDropzone = document.getElementById('videoDropzone');
  const videoFileInput = document.getElementById('videoFileInput');
  const btnBrowseFile = document.getElementById('btnBrowseFile');
  const importSamplesWrap = document.getElementById('importSamplesWrap');
  const importWorkspace = document.getElementById('importWorkspace');
  const importVideoPlayer = document.getElementById('importVideoPlayer');
  const importFileNameText = document.getElementById('importFileNameText');
  const importFileSizeBadge = document.getElementById('importFileSizeBadge');
  const btnChangeVideo = document.getElementById('btnChangeVideo');
  const btnStartVideoRecognize = document.getElementById('btnStartVideoRecognize');
  const importStandbyBox = document.getElementById('importStandbyBox');
  const importAnalyzingBox = document.getElementById('importAnalyzingBox');
  const importAnalyzingStepText = document.getElementById('importAnalyzingStepText');
  const importResultBox = document.getElementById('importResultBox');
  const importStatusBadge = document.getElementById('importStatusBadge');
  const importTimeBadge = document.getElementById('importTimeBadge');
  const importResultIcon = document.getElementById('importResultIcon');
  const importResultLabel = document.getElementById('importResultLabel');
  const importResultCat = document.getElementById('importResultCat');
  const importConfidenceText = document.getElementById('importConfidenceText');
  const importConfidenceBar = document.getElementById('importConfidenceBar');
  const btnImportSpeak = document.getElementById('btnImportSpeak');
  const btnImportAddToSentence = document.getElementById('btnImportAddToSentence');
  const importErrorBox = document.getElementById('importErrorBox');
  const importErrorMessage = document.getElementById('importErrorMessage');
  const importErrorHint = document.getElementById('importErrorHint');
  const btnRetryImport = document.getElementById('btnRetryImport');

  let currentImportFile = null;
  let currentImportFileName = '';
  let currentVideoObjectUrl = null;
  let lastRecognizedVideoLabel = '';

  function switchTranslateMode(mode) {
    if (mode === 'import') {
      tabModeImportVideo?.classList.add('active');
      tabModeLiveCam?.classList.remove('active');
      if (practiceSelectorBar) practiceSelectorBar.style.setProperty('display', 'none', 'important');
      if (practiceStudioBox) practiceStudioBox.style.display = 'none';
      if (videoImportStudio) videoImportStudio.style.display = 'block';

      // Nếu webcam đang chạy thì tạm dừng để tiết kiệm tài nguyên
      if (state.cameraEnabled) {
        stopClientWebcam();
      }
    } else {
      tabModeLiveCam?.classList.add('active');
      tabModeImportVideo?.classList.remove('active');
      if (practiceSelectorBar) practiceSelectorBar.style.removeProperty('display');
      if (practiceStudioBox) practiceStudioBox.style.display = '';
      if (videoImportStudio) videoImportStudio.style.display = 'none';
    }
  }

  tabModeLiveCam?.addEventListener('click', () => switchTranslateMode('live'));
  tabModeImportVideo?.addEventListener('click', () => switchTranslateMode('import'));

  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  let isTranscoding = false;

  async function transcodeToWebH264(file) {
    if (!file || isTranscoding) return;
    isTranscoding = true;
    const codecNotice = document.getElementById('importCodecNotice');
    if (codecNotice) codecNotice.style.display = 'flex';

    try {
      const formData = new FormData();
      formData.append('file', file, file.name || 'video.mp4');
      const resp = await fetch(`${API_BASE}/api/convert_video_preview`, {
        method: 'POST',
        body: formData
      });
      if (resp.ok) {
        const blob = await resp.blob();
        if (currentVideoObjectUrl) URL.revokeObjectURL(currentVideoObjectUrl);
        currentVideoObjectUrl = URL.createObjectURL(blob);
        if (importVideoPlayer) {
          importVideoPlayer.src = currentVideoObjectUrl;
          importVideoPlayer.currentTime = 0;
          importVideoPlayer.load();
          importVideoPlayer.play().catch(() => {});
        }
      }
    } catch (err) {
      console.warn('Lỗi tự động chuyển mã preview video:', err);
    } finally {
      if (codecNotice) codecNotice.style.display = 'none';
      isTranscoding = false;
    }
  }

  function loadVideoFile(file, fileName) {
    if (!file) return;
    currentImportFile = file;
    currentImportFileName = fileName || file.name || 'video.mp4';

    if (currentVideoObjectUrl) {
      URL.revokeObjectURL(currentVideoObjectUrl);
    }
    currentVideoObjectUrl = URL.createObjectURL(file);

    const codecNotice = document.getElementById('importCodecNotice');
    if (codecNotice) codecNotice.style.display = 'none';

    if (importVideoPlayer) {
      importVideoPlayer.src = currentVideoObjectUrl;
      importVideoPlayer.currentTime = 0;
      importVideoPlayer.load();
    }

    // Nếu là video thu từ OpenCV (thường có _raw.mp4, _skeleton.mp4 hoặc gắn nhãn REJECTED/ACCEPTED)
    const isOpencvClip = /(_raw|_skeleton)\.mp4$/i.test(file.name) || /_(REJECTED|ACCEPTED)_/i.test(file.name);
    if (isOpencvClip) {
      transcodeToWebH264(file);
    } else {
      setTimeout(() => {
        if (importVideoPlayer && importVideoPlayer.videoWidth === 0 && currentImportFile && !isTranscoding) {
          transcodeToWebH264(currentImportFile);
        }
      }, 700);
    }

    if (importFileNameText) importFileNameText.textContent = currentImportFileName;
    if (importFileSizeBadge) importFileSizeBadge.textContent = file.size ? formatBytes(file.size) : 'Video file';

    if (videoDropzone) videoDropzone.style.display = 'none';
    if (importWorkspace) importWorkspace.style.display = 'block';

    // Đưa các box trạng thái về ban đầu
    if (importStandbyBox) importStandbyBox.style.display = 'flex';
    if (importAnalyzingBox) importAnalyzingBox.style.display = 'none';
    if (importResultBox) importResultBox.style.display = 'none';
    if (importErrorBox) importErrorBox.style.display = 'none';
  }

  // Kéo và thả file video
  if (videoDropzone) {
    videoDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      videoDropzone.classList.add('dragover');
    });
    videoDropzone.addEventListener('dragleave', () => {
      videoDropzone.classList.remove('dragover');
    });
    videoDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      videoDropzone.classList.remove('dragover');
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        loadVideoFile(file, file.name);
      }
    });
    videoDropzone.addEventListener('click', (e) => {
      if (e.target.closest('#btnBrowseFile') || e.target === videoDropzone || videoDropzone.contains(e.target)) {
        videoFileInput?.click();
      }
    });
  }

  btnBrowseFile?.addEventListener('click', (e) => {
    e.stopPropagation();
    videoFileInput?.click();
  });

  videoFileInput?.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      loadVideoFile(file, file.name);
      videoFileInput.value = '';
    }
  });

  // Chọn video mẫu thử nhanh
  importSamplesWrap?.addEventListener('click', async (e) => {
    const btn = e.target.closest('.btn-sample-chip');
    if (!btn) return;
    const sampleId = btn.dataset.sampleId;
    if (!sampleId) return;

    document.querySelectorAll('.btn-sample-chip').forEach(b => b.classList.toggle('active', b === btn));

    const g = GESTURE_DATA.find(item => item.id === sampleId);
    const videoUrl = g ? g.video : `tutorials/${sampleId}.mp4`;
    const label = g ? g.name : sampleId;

    try {
      const res = await fetch(videoUrl);
      if (!res.ok) throw new Error('Không thể tải file mẫu');
      const blob = await res.blob();
      const sampleFile = new File([blob], `${sampleId}.mp4`, { type: 'video/mp4' });
      loadVideoFile(sampleFile, `Mẫu chuẩn: ${label} (${sampleId}.mp4)`);
    } catch (err) {
      console.warn('Fallback tải video mẫu trực tiếp vào player:', err);
      if (importVideoPlayer) {
        importVideoPlayer.src = videoUrl;
        importVideoPlayer.load();
      }
      if (importFileNameText) importFileNameText.textContent = `Mẫu chuẩn: ${label} (${sampleId}.mp4)`;
      if (importFileSizeBadge) importFileSizeBadge.textContent = 'Mẫu chuẩn VSLR';
      if (videoDropzone) videoDropzone.style.display = 'none';
      if (importWorkspace) importWorkspace.style.display = 'block';
    }
  });

  function resetImportDropzone() {
    if (currentVideoObjectUrl) {
      URL.revokeObjectURL(currentVideoObjectUrl);
      currentVideoObjectUrl = null;
    }
    currentImportFile = null;
    currentImportFileName = '';
    if (importVideoPlayer) {
      importVideoPlayer.pause();
      importVideoPlayer.removeAttribute('src');
    }
    if (importWorkspace) importWorkspace.style.display = 'none';
    if (videoDropzone) videoDropzone.style.display = 'block';
    const codecNotice = document.getElementById('importCodecNotice');
    if (codecNotice) codecNotice.style.display = 'none';
    document.querySelectorAll('.btn-sample-chip').forEach(b => b.classList.remove('active'));
  }

  importVideoPlayer?.addEventListener('error', () => {
    if (currentImportFile && !isTranscoding) {
      transcodeToWebH264(currentImportFile);
    }
  });

  importVideoPlayer?.addEventListener('loadeddata', () => {
    const codecNotice = document.getElementById('importCodecNotice');
    if (codecNotice) codecNotice.style.display = 'none';
  });

  btnChangeVideo?.addEventListener('click', resetImportDropzone);
  btnRetryImport?.addEventListener('click', resetImportDropzone);

  // Nhận diện Video
  btnStartVideoRecognize?.addEventListener('click', async () => {
    if (!currentImportFile) {
      if (importVideoPlayer && importVideoPlayer.src) {
        try {
          const resp = await fetch(importVideoPlayer.src);
          const b = await resp.blob();
          currentImportFile = new File([b], 'sample.mp4', { type: 'video/mp4' });
        } catch (e) {
          alert('Vui lòng chọn hoặc tải lên một file video.');
          return;
        }
      } else {
        alert('Vui lòng chọn hoặc tải lên một file video.');
        return;
      }
    }

    if (btnStartVideoRecognize) btnStartVideoRecognize.disabled = true;

    if (importStandbyBox) importStandbyBox.style.display = 'none';
    if (importResultBox) importResultBox.style.display = 'none';
    if (importErrorBox) importErrorBox.style.display = 'none';
    if (importAnalyzingBox) importAnalyzingBox.style.display = 'flex';

    if (importAnalyzingStepText) {
      importAnalyzingStepText.textContent = '1/2: Trích xuất 203 đặc trưng MediaPipe Holistic v3...';
    }
    const stepTimer = setTimeout(() => {
      if (importAnalyzingStepText) {
        importAnalyzingStepText.textContent = '2/2: Suy luận mạng nơ-ron BiLSTM (24 nhãn VSLR)...';
      }
    }, 900);

    const startTime = performance.now();

    try {
      const formData = new FormData();
      formData.append('file', currentImportFile, currentImportFileName || 'video.mp4');
      formData.append('signer', state.signerName || 'Khách');
      formData.append('session_id', state.sessionId);
      formData.append('add_to_sentence', 'false');

      const response = await fetch(`${API_BASE}/api/recognize_video`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      clearTimeout(stepTimer);
      const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(1);

      if (importAnalyzingBox) importAnalyzingBox.style.display = 'none';

      if (!response.ok || data.status === 'error') {
        if (importErrorBox) importErrorBox.style.display = 'flex';
        if (importErrorMessage) {
          importErrorMessage.textContent = data.detail || data.message || 'Lỗi khi nhận diện video.';
        }
        if (importErrorHint && data.hint) {
          importErrorHint.innerHTML = `💡 <strong>Mẹo:</strong> ${data.hint}`;
        }
        return;
      }

      lastRecognizedVideoLabel = data.label;
      if (importResultBox) importResultBox.style.display = 'flex';

      const matchedGesture = GESTURE_DATA.find(g =>
        g.name.toLowerCase().trim() === data.label.toLowerCase().trim()
      );

      if (importResultIcon) importResultIcon.textContent = matchedGesture ? matchedGesture.icon : '✋';
      if (importResultLabel) importResultLabel.textContent = data.label;
      if (importResultCat) {
        const cat = matchedGesture ? matchedGesture.cat : 'Giao tiếp';
        const dur = data.meta && data.meta.duration ? `${data.meta.duration}s` : '1.5s';
        importResultCat.textContent = `Chủ đề: ${cat} • Thời lượng: ${dur} • 24 Nhãn chuẩn VSLR`;
      }

      if (importStatusBadge) {
        if (data.accepted) {
          importStatusBadge.className = 'result-status-badge';
          importStatusBadge.textContent = '✓ Nhận diện thành công';
        } else {
          importStatusBadge.className = 'result-status-badge rejected';
          importStatusBadge.textContent = `⚠️ Nghi ngờ (${data.reason || 'Độ tin cậy thấp'})`;
        }
      }

      if (importTimeBadge) importTimeBadge.textContent = `⏱ ${elapsedSec}s`;

      if (importConfidenceText) importConfidenceText.textContent = `${data.confidence}%`;
      if (importConfidenceBar) {
        importConfidenceBar.style.width = `${Math.min(100, Math.max(5, data.confidence))}%`;
        importConfidenceBar.style.background = data.confidence >= 70
          ? 'linear-gradient(90deg, #10b981, #059669)'
          : 'linear-gradient(90deg, #f59e0b, #d97706)';
      }

      // Phát âm ngay trên loa thiết bị
      speakText(data.label);

      // Cập nhật dải lịch sử nhận diện gần đây
      pushRecentGesture(data.label, Math.round(data.confidence));

    } catch (err) {
      clearTimeout(stepTimer);
      if (importAnalyzingBox) importAnalyzingBox.style.display = 'none';
      if (importErrorBox) importErrorBox.style.display = 'flex';
      if (importErrorMessage) {
        importErrorMessage.textContent = `Không thể kết nối tới server AI: ${err.message}`;
      }
    } finally {
      if (btnStartVideoRecognize) btnStartVideoRecognize.disabled = false;
    }
  });

  btnImportSpeak?.addEventListener('click', () => {
    if (lastRecognizedVideoLabel) {
      lastSpokenTime = 0;
      speakText(lastRecognizedVideoLabel);
    }
  });

  btnImportAddToSentence?.addEventListener('click', () => {
    if (lastRecognizedVideoLabel) {
      state.sentence.push(lastRecognizedVideoLabel);
      renderSentence();
      if (btnImportAddToSentence) {
        const origText = btnImportAddToSentence.textContent;
        btnImportAddToSentence.textContent = '✓ Đã thêm vào câu!';
        btnImportAddToSentence.style.background = '#ecfdf5';
        btnImportAddToSentence.style.color = '#059669';
        setTimeout(() => {
          btnImportAddToSentence.textContent = origText;
          btnImportAddToSentence.style.background = '';
          btnImportAddToSentence.style.color = '';
        }, 1500);
      }
    }
  });

  updateCameraStateUI();
  initEventStream();
  startAutoDiscovery();

});
