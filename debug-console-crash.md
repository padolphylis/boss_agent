# Debug Session: console-crash

Status: [OPEN]

## Symptom

The local `zhipin.html` page closes, refreshes, or exits after the browser developer tools console is opened.

## Hypotheses

1. A remote security script detects developer tools and forces a reload or navigation.
2. An uncaught JavaScript exception occurs during the developer-tools state change.
3. A remote script request fails or triggers a risk-control response when developer tools are opened.
4. A browser extension or automation environment closes the page independently of the document.

## Evidence

No runtime evidence collected yet.

