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
  ];

  // =========================================================================
  // 2. APPLICATION STATE
  // =========================================================================
  const state = {
    activeView: 'view-home',
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
  const recentGesturesHistory = [];

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
      ensureLiveStreamRunning();
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

    // 1. Gọi API VieNeu-TTS backend (phát trực tiếp ra loa máy tính & trả về file âm thanh)
    fetch(`/api/tts?text=${encodeURIComponent(cleanText)}&voice=${encodeURIComponent(state.ttsVoice)}&play_server=true`)
      .then(res => {
        if (res.headers.get("content-type")?.includes("audio/wav")) {
          return res.blob();
        }
        return null;
      })
      .then(blob => {
        if (blob) {
          const audioUrl = URL.createObjectURL(blob);
          const audio = new Audio(audioUrl);
          audio.play().catch(() => {});
        }
      })
      .catch(() => {
        // Dự phòng: Nếu server ngắt kết nối thì dùng tạm SpeechSynthesis trình duyệt
        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(cleanText);
          utterance.lang = 'vi-VN';
          window.speechSynthesis.speak(utterance);
        }
      });
  }

  // =========================================================================
  // 5. BACKEND REST API ACTIONS (SPACE, CLEAR, SPEAK, TOGGLE)
  // =========================================================================
  async function triggerBackendAction(actionName) {
    try {
      const res = await fetch(`/api/action?action=${encodeURIComponent(actionName)}`, {
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

  // Phím bấm giao diện
  btnMainToggleTranslate?.addEventListener('click', () => triggerBackendAction('space'));
  document.getElementById('btnClearSentence')?.addEventListener('click', () => triggerBackendAction('clear'));
  document.getElementById('btnSpeakSentence')?.addEventListener('click', () => triggerBackendAction('speak'));
  btnToggleHands?.addEventListener('click', () => triggerBackendAction('toggle_hands'));
  toggleModeBtn?.addEventListener('click', () => triggerBackendAction('toggle_mode'));

  // Phím tắt bàn phím (Tương thích 100% với CHAY_CAMERA_*.bat)
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.code === 'Space') {
      e.preventDefault();
      if (state.activeView !== 'view-translate') {
        navigateToView('view-translate');
      }
      triggerBackendAction('space');
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

    state.eventSource = new EventSource('/api/events');

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
        state.recMode = data.rec_mode || 'auto';
        state.showHands = !!data.show_hands;
        state.ttsVoice = data.tts_voice || 'Trúc Ly';
        state.sentence = data.sentence || [];
        updateModeUI();
        updateHandsUI();
        renderSentence();
        if (hudVoiceStatus) hudVoiceStatus.textContent = `${state.ttsVoice} (48kHz)`;
        if (hudCamStatus && data.camera) hudCamStatus.textContent = data.camera;
        break;

      case 'telemetry':
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

      case 'prediction':
        if (data.accepted) {
          // Hiển thị chữ dự đoán lớn kèm hiệu ứng pop
          if (heroPredictedWord) {
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
      fetch(`/api/action?action=clear`, { method: 'POST' }).then(() => {
        state.sentence.forEach(w => {
          // preserve client edit
        });
      });
    }
  });

  // =========================================================================
  // 7. CAMERA STREAM
  // =========================================================================
  function ensureLiveStreamRunning() {
    if (liveStreamImg) {
      liveStreamImg.src = '/api/video_feed?t=' + Date.now();
      camPrompt?.classList.add('hidden');
    }
  }

  btnStartRealCam?.addEventListener('click', () => {
    ensureLiveStreamRunning();
  });

  // Handle stream reload if image errors
  liveStreamImg?.addEventListener('error', () => {
    camPrompt?.classList.remove('hidden');
    setTimeout(ensureLiveStreamRunning, 2000);
  });

  // =========================================================================
  // 8. DICTIONARY RENDERING & SEARCH (24 Gestures)
  // =========================================================================
  const dictGridContainer = document.getElementById('dictGridContainer');
  const dictFilterInput = document.getElementById('dictFilterInput');

  function renderDictionary(filter = '') {
    if (!dictGridContainer) return;

    const filtered = GESTURE_DATA.filter(g => 
      g.name.toLowerCase().includes(filter.toLowerCase()) ||
      g.cat.toLowerCase().includes(filter.toLowerCase())
    );

    dictGridContainer.innerHTML = filtered.map(g => `
      <div class="dict-card" data-gesture-id="${g.id}">
        <div class="dict-card-left">
          <span class="dict-icon">${g.icon}</span>
          <div>
            <div class="dict-name">${g.name}</div>
            <div class="dict-cat">${g.cat}</div>
          </div>
        </div>
        <span class="dict-arrow">→</span>
      </div>
    `).join('');
  }

  renderDictionary();

  dictFilterInput?.addEventListener('input', (e) => {
    renderDictionary(e.target.value.trim());
  });

  // =========================================================================
  // 9. MODALS (ABOUT & GESTURE DETAIL)
  // =========================================================================
  const aboutModal = document.getElementById('aboutModal');
  const gestureDetailModal = document.getElementById('gestureDetailModal');

  document.getElementById('openAboutBtn')?.addEventListener('click', () => {
    aboutModal?.classList.add('open');
  });

  document.getElementById('closeAboutModal')?.addEventListener('click', () => {
    aboutModal?.classList.remove('open');
  });

  document.getElementById('closeAboutBtnBottom')?.addEventListener('click', () => {
    aboutModal?.classList.remove('open');
  });

  // Chi tiết cử chỉ
  let activeModalGesture = null;
  const detailCatBadge = document.getElementById('detailCatBadge');
  const detailGestureTitle = document.getElementById('detailGestureTitle');
  const detailGestureIcon = document.getElementById('detailGestureIcon');
  const detailGestureDesc = document.getElementById('detailGestureDesc');
  const detailGestureTip = document.getElementById('detailGestureTip');

  function openGestureDetail(g) {
    activeModalGesture = g;
    if (detailCatBadge) detailCatBadge.textContent = g.cat;
    if (detailGestureTitle) detailGestureTitle.textContent = g.name;
    if (detailGestureIcon) detailGestureIcon.textContent = g.icon;
    if (detailGestureDesc) detailGestureDesc.textContent = g.desc;
    if (detailGestureTip) detailGestureTip.textContent = g.tip;
    gestureDetailModal?.classList.add('open');
  }

  document.getElementById('closeDetailModal')?.addEventListener('click', () => {
    gestureDetailModal?.classList.remove('open');
    activeModalGesture = null;
  });

  document.addEventListener('click', (e) => {
    const card = e.target.closest('.dict-card');
    if (card) {
      const gid = card.dataset.gestureId;
      const found = GESTURE_DATA.find(g => g.id === gid);
      if (found) openGestureDetail(found);
    }
  });

  // Phát âm chuẩn VieNeu-TTS Trúc Ly từ trong Modal Từ điển
  document.getElementById('btnDetailSpeak')?.addEventListener('click', () => {
    if (activeModalGesture) speakText(activeModalGesture.name);
  });

  document.getElementById('btnDetailPractice')?.addEventListener('click', () => {
    gestureDetailModal?.classList.remove('open');
    navigateToView('view-translate');
  });

  // Đóng modal khi click ra nền ngoài hoặc nhấn ESC
  [aboutModal, gestureDetailModal].forEach(m => {
    m?.addEventListener('click', (e) => {
      if (e.target === m) m.classList.remove('open');
    });
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      aboutModal?.classList.remove('open');
      gestureDetailModal?.classList.remove('open');
    }
  });

  // =========================================================================
  // 10. INITIALIZATION
  // =========================================================================
  initEventStream();
  ensureLiveStreamRunning();

});
