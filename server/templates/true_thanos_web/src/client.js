// get DOM elements
var iceConnectionLog = {textContent: ''},
    iceGatheringLog = {textContent: ''},
    signalingLog = {textContent: ''},
    video = {srcObject: ''},
    return_answer;

// peer connection
var pc = null;

// data channel
var dc = null;

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    var pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function() {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
        console.log('iceGatheringLog: ' + iceGatheringLog.textContent);
    }, false);
    iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function() {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
        console.log('iceConnectionLog: ' + iceConnectionLog.textContent);
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function() {
        signalingLog.textContent += ' -> ' + pc.signalingState;
        console.log('signalingLog: ' + signalingLog.textContent);
    }, false);
    signalingLog.textContent = pc.signalingState;

    // connect audio / video
    pc.addEventListener('track', function(evt) {
        video.srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;

        return fetch('http://localhost:5000/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: ''
            }),
            /*
            headers: {
                'Content-Type': 'application/json'
            },
            */
            method: 'POST'
        });
        /*
        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'http://127.0.0.1:5000/offer', false);
        // xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: document.getElementById('video-transform').value
            }),
        );
        return xhr.responseText
        */
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        //document.getElementById('answer-sdp').textContent = answer.sdp;
        return_answer = answer;
        console.log('Return Answer SDP: ' + return_answer.sdp);
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}
function start(){
    console.log('Start Create');
    pc = createPeerConnection();

    var constraints = {
        audio: false,
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        stream.getTracks().forEach(function(track) {
            pc.addTrack(track, stream);
        });
        return negotiate();
        }, function(err) {
        alert('Could not acquire media: ' + err);
    });
}
