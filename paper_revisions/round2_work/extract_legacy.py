"""Read-only extraction of Word binary streams; preserve source hashes."""
import struct, json, hashlib
from pathlib import Path
U16=lambda b,o:struct.unpack_from('<H',b,o)[0]
U32=lambda b,o:struct.unpack_from('<I',b,o)[0]
def streams(path):
    b=path.read_bytes(); size=1<<U16(b,30)
    sec=lambda n:b[(n+1)*size:(n+2)*size]
    dif=list(struct.unpack_from('<109I',b,76)); n=U32(b,68)
    for _ in range(U32(b,72)):
        a=struct.unpack('<'+'I'*(size//4),sec(n)); dif.extend(a[:-1]); n=a[-1]
    fat=[]
    for n in dif:
        if n<0xfffffffa:fat.extend(struct.unpack('<'+'I'*(size//4),sec(n)))
    def chain(n,table,reader):
        out=[]; seen=set()
        while n<0xfffffffa:
            if n in seen:raise ValueError('cyclic stream')
            seen.add(n); out.append(reader(n)); n=table[n]
        return b''.join(out)
    directory=chain(U32(b,48),fat,sec); entries={}
    for i in range(0,len(directory),128):
        a=directory[i:i+128]; ln=U16(a,64)
        if ln:entries[a[:ln-2].decode('utf-16le')]=(a[66],U32(a,116),U32(a,120))
    root=entries['Root Entry']; mini=chain(root[1],fat,sec)[:root[2]]
    mf=chain(U32(b,60),fat,sec); mf=list(struct.unpack('<'+'I'*(len(mf)//4),mf))
    out={}
    for k,(kind,n,length) in entries.items():
        if kind!=2:continue
        out[k]=(chain(n,mf,lambda i:mini[i*64:(i+1)*64]) if length<4096 else chain(n,fat,sec))[:length]
    return out
def extract(path):
    st=streams(path); w=st['WordDocument']; fib=U16(w,2)
    meta={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nFib':fib,'streams':list(st)}
    # Word 6/95 has contiguous single-byte text between fcMin and fcMac.
    if fib<0xc1:
        text=w[U32(w,24):U32(w,28)].decode('cp1252',errors='replace')
        meta['extraction']='Word 6/95 fcMin:fcMac cp1252'
    else:
        off=32; off+=2+2*U16(w,off); off+=2+4*U16(w,off); off+=2
        fc,lcb=struct.unpack_from('<II',w,off+33*8)
        table=st['1Table' if U16(w,10)&0x200 else '0Table']; clx=table[fc:fc+lcb]; pos=0
        while clx[pos]==1:pos+=3+U16(clx,pos+1)
        if clx[pos]!=2:raise ValueError('missing piece table')
        length=U32(clx,pos+1); plc=clx[pos+5:pos+5+length]; count=(length-4)//12
        cps=struct.unpack_from('<'+'I'*(count+1),plc); parts=[]
        for i in range(count):
            loc=U32(plc,4*(count+1)+8*i+2); compressed=bool(loc&0x40000000); loc&=0x3fffffff
            if compressed:loc//=2
            chars=cps[i+1]-cps[i]; width=1 if compressed else 2
            parts.append(w[loc:loc+chars*width].decode('cp1252' if compressed else 'utf-16le',errors='replace'))
        text=''.join(parts); meta['extraction']='Word piece table'
    return text.replace('\r','\n').replace('\x07',' | '),meta
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    for p in root.parent.glob('*.doc'):
        text,meta=extract(p); (root/(p.stem+'.txt')).write_text(text,encoding='utf8'); (root/(p.stem+'.source.json')).write_text(json.dumps(meta,indent=2),encoding='utf8'); print(meta, len(text))
