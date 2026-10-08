// Package certs generates and caches a self-signed TLS certificate for the
// media edge. iOS Safari (and modern browsers generally) refuse
// getUserMedia on a plain-HTTP, non-localhost origin, so the edge must serve
// over HTTPS even for a same-network phone.
package certs

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"
)

const validity = 825 * 24 * time.Hour // Apple's max lifetime for a leaf cert

// Ensure returns paths to a cert/key pair in dir valid for the given IPs,
// generating a new self-signed one if none exists yet or the IPs changed.
func Ensure(dir string, ips []string) (certFile, keyFile string, err error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", err
	}
	certFile = filepath.Join(dir, "media-edge.crt")
	keyFile = filepath.Join(dir, "media-edge.key")

	if certCoversIPs(certFile, ips) {
		return certFile, keyFile, nil
	}
	if err := generate(certFile, keyFile, ips); err != nil {
		return "", "", err
	}
	return certFile, keyFile, nil
}

func certCoversIPs(certFile string, ips []string) bool {
	data, err := os.ReadFile(certFile)
	if err != nil {
		return false
	}
	block, _ := pem.Decode(data)
	if block == nil {
		return false
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil || time.Now().After(cert.NotAfter) {
		return false
	}
	have := make(map[string]bool, len(cert.IPAddresses))
	for _, ip := range cert.IPAddresses {
		have[ip.String()] = true
	}
	for _, ip := range ips {
		if !have[ip] {
			return false
		}
	}
	return true
}

func generate(certFile, keyFile string, ips []string) error {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return fmt.Errorf("generate key: %w", err)
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return err
	}

	tmpl := &x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{Organization: []string{"inference-orchestrator media edge (self-signed)"}},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(validity),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true, // self-signed leaf acting as its own CA
		DNSNames:              []string{"localhost"},
	}
	for _, ip := range ips {
		if parsed := net.ParseIP(ip); parsed != nil {
			tmpl.IPAddresses = append(tmpl.IPAddresses, parsed)
		}
	}
	tmpl.IPAddresses = append(tmpl.IPAddresses, net.ParseIP("127.0.0.1"))

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return fmt.Errorf("create certificate: %w", err)
	}

	certOut, err := os.Create(certFile)
	if err != nil {
		return err
	}
	defer certOut.Close()
	if err := pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: der}); err != nil {
		return err
	}

	keyBytes, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return err
	}
	keyOut, err := os.OpenFile(keyFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer keyOut.Close()
	return pem.Encode(keyOut, &pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes})
}
