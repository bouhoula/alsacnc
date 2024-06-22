function detectCMPTCF() {
    methodNames = ["__tcfapi", "__cmp"]
    let data = {}
    for (let methodName of methodNames) {
        if (typeof window[methodName] === 'function') {
			data = {}
            try {
                window[methodName]('getTCData', 2, function (tcdata, success) {
                    data = tcdata;
                });
            } catch (e) {}

			if (typeof data === 'object' && data !== null) {
                data.method_name = methodName;
            } else {
                data = {"method_name": methodName};
            }
            if (Object.keys(data).length > 1) {
                return data
            }
        }
    }
    return data;
}

return detectCMPTCF()